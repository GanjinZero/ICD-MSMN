import torch
from torch import nn
import torch.nn.functional as F

from model.text_encoder import TextEncoder
from model.decoder import create_decoder
from model.label_encoder import LabelEncoder
from model.losses import loss_fn
from evaluation import all_metrics
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class IcdModel(nn.Module):
    def __init__(self, word_config={}, combine_config={},
                 decoder_config={}, label_config={}, loss_config={}, args=None):
        super().__init__()
        self.encoder = TextEncoder(word_config, combine_config)
        self.decoder = create_decoder(decoder_config)
        self.label_encoder = LabelEncoder(label_config)
        self.loss_config = loss_config
        self.args = args
        
    def calculate_text_hidden(self, input_word, word_mask):
        hidden = self.encoder(input_word, word_mask)
        return hidden
    
    def calculate_label_hidden(self):
        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        self.label_feats = self.label_encoder(label_hidden, self.c_word_mask)

    def forward(self, batch, rdrop=False):
        if rdrop:
            return self.forward_rdrop(batch)
        input_word, word_mask = batch[0:2]
        hidden = self.calculate_text_hidden(input_word, word_mask)
        # mc_logits = self.decoder(hidden, word_mask)
        
        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        label_feats = self.label_encoder(label_hidden, self.c_word_mask)
        c_logits = self.decoder(hidden, word_mask, label_feats)
        
        # mc_label = batch[-1]
        c_label = batch[-2]
        # mc_loss = loss_fn(mc_logits, mc_label, self.loss_config)
        mc_loss = 0.0
        
        c_loss = loss_fn(c_logits, c_label, self.loss_config)
        loss = mc_loss * self.loss_config['main_code_loss_weight'] + \
               c_loss * self.loss_config['code_loss_weight']
        return {'mc_loss':mc_loss, 'c_loss':c_loss, 'loss':loss}
               
    
    def forward_rdrop(self, batch):
        input_word, word_mask = batch[0:2]
        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        label_feats = self.label_encoder(label_hidden, self.c_word_mask)
        
        hidden0 = self.calculate_text_hidden(input_word, word_mask)
        hidden1 = self.calculate_text_hidden(input_word, word_mask)
        
        # ignore mc_logits
        c_logits0 = self.decoder(hidden0, word_mask, label_feats)
        c_logits1 = self.decoder(hidden1, word_mask, label_feats)
        
        c_label = batch[-2]
        
        c_loss = (loss_fn(c_logits0, c_label, self.loss_config) + \
                  loss_fn(c_logits1, c_label, self.loss_config)) * 0.5
        
        kl_loss = compute_kl_loss(c_logits0, c_logits1)
        
        loss = self.loss_config['rdrop_alpha'] * kl_loss + \
               c_loss * self.loss_config['code_loss_weight']
        return {'kl_loss':kl_loss, 'c_loss':c_loss, 'loss':loss}

    def predict(self, batch, threshold=None):
        input_word, word_mask = batch[0:2]
        hidden = self.calculate_text_hidden(input_word, word_mask)
        assert hasattr(self, 'label_feats')
        yhat_raw = self.decoder(hidden, word_mask, self.label_feats)
        if isinstance(yhat_raw, tuple):
            yhat_raw = yhat_raw[0]
        if threshold is None:
            threshold = self.args.prob_threshold
        yhat = yhat_raw >= threshold
        y = batch[-2]
        return {"yhat_raw": yhat_raw, "yhat": yhat, "y": y}

    def configure_optimizers(self, train_dataloader=None):
        args = self.args
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "AdamW":
            no_decay = ["bias", "LayerNorm.weight"]
            params = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                    "lr": args.learning_rate
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": args.learning_rate
                }, 
            ]
            
            optimizer = AdamW(params, eps=args.adam_epsilon)
            
            self.total_steps = len(train_dataloader) * args.train_epoch
            
            if not hasattr(self.args, 'scheduler') or self.args.scheduler == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            elif self.args.scheduler == "constant":
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio)
                )
            elif self.args.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            return [optimizer], [scheduler]
