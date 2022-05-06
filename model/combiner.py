import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from reformer_pytorch import Reformer
from model.pos import AbsPositionEmbedding


class Combiner(nn.Module):
    def __init__(self, combine_config={}):
        super(Combiner, self).__init__()
        self.combine_config = combine_config
        self.input_dim = self.combine_config['input_dim']
        self.output_dim = self.combine_config['dim']
        
    def lstm_forward(self, x, lengths, lstm):
        np_lengths = lengths.cpu().numpy()
        np_lengths[np_lengths==0] = 1
        x_pack = pack_padded_sequence(x, np_lengths, batch_first=True, enforce_sorted=False)
        h_pack, _ = lstm(x_pack)
        h, _ = pad_packed_sequence(h_pack, batch_first=True)
        return h
    
    def combine_forward(self, embeds, word_mask):
        raise NotImplementedError
        
    def forward(self, word_hidden, word_mask):
        embeds = self.combine_forward(word_hidden, word_mask)
        if hasattr(self, 'reduce_linear'):
            embeds = self.reduce_linear(embeds)
        return embeds

class Naive_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(Naive_Combiner, self).__init__(combine_config)
        if self.input_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.input_dim, self.output_dim)
            
    def combine_forward(self, embeds, word_mask):
        return embeds
        
        
class LSTM_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(LSTM_Combiner, self).__init__(combine_config)
        self.rnn_dim = self.combine_config['rnn_dim']
        self.num_layers = self.combine_config['num_layers']
        self.combine_lstm_dropout = self.combine_config['lstm_dropout']
        self.combine_lstm = nn.LSTM(self.input_dim,
                                 self.rnn_dim // 2,
                                 self.num_layers,
                                 bidirectional=True,
                                 dropout=self.combine_lstm_dropout,
                                 batch_first=True)
        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)
    
    def combine_forward(self, embeds, word_mask):
        word_lengths = torch.sum(word_mask, dim=1)
        h = self.lstm_forward(embeds, word_lengths, self.combine_lstm)
        return h
    
    
class Reformer_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(Reformer_Combiner, self).__init__(combine_config)

        self.rnn_dim = self.combine_config['rnn_dim']
        self.num_layers = self.combine_config['num_layers']
        self.reformer_head = self.combine_config['reformer_head']
        self.n_hashes = self.combine_config['n_hashes']
        self.linear_reformer = nn.Linear(self.input_dim, self.rnn_dim)
        self.pos_emb = AbsPositionEmbedding(self.combine_config)
        self.local_attention_head = self.combine_config['local_attention_head']
        self.pkm_layers = self.combine_config['pkm_layers']
        self.dropout = nn.Dropout(self.combine_config['transformer_dropout'])
        
        if self.combine_config['layer_norm']:
            self.norm = nn.LayerNorm(self.rnn_dim)
        
        self.reformer = Reformer(
            dim = self.rnn_dim,
            depth = self.num_layers,
            heads = self.reformer_head,
            n_hashes = self.n_hashes,
            lsh_dropout = 0.1,
            n_local_attn_heads = self.local_attention_head,
            pkm_layers = self.pkm_layers,
            causal = False)
        
        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)

    def combine_forward(self, embeds, word_mask):
        x_embed = self.linear_reformer(embeds)
        x_embed = self.pos_emb(x_embed) + x_embed
        if self.combine_config['layer_norm']:
            x_embed = self.norm(x_embed)
        x_embed = self.dropout(x_embed)
        word_hidden = self.reformer(x_embed, i_mask=word_mask)
        return word_hidden  
        

def create_combiner(combine_config):
    if combine_config['model'] == 'naive':
        combiner = Naive_Combiner(combine_config)
    if combine_config['model'] == 'lstm':
        combiner = LSTM_Combiner(combine_config)
    if combine_config['model'] == 'reformer':
        combiner = Reformer_Combiner(combine_config)
    return combiner
