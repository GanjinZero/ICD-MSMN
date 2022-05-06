import torch
from torch import nn
from model.mlp import MLP


class LabelEncoder(nn.Module):
    def __init__(self, label_config={}):
        super(LabelEncoder, self).__init__()
        self.label_config = label_config
        self.pooling = self.label_config['pooling']
        if label_config['num_layers'] > 0:
            self.mlp = MLP(input_dim=label_config['input_dim'],
                           hidden_dim=label_config['input_dim'],
                           output_dim=label_config['input_dim'],
                           num_layers=label_config['num_layers'],
                           dropout=label_config['dropout'])

    def forward(self, input_feat, word_mask=None):
        # input_feat: Label * Seq * Hidden
        if not hasattr(self, 'pooling') or self.pooling == 'max':
            input_feat = input_feat.max(dim=1)[0]  # Label * Hidden
        elif self.pooling == 'mean':
            input_feat = input_feat.mean(dim=1)
        elif self.pooling == 'last':
            assert word_mask is not None
            word_length = word_mask.sum(-1)
            word_length[word_length==0] = 1
            idx = (word_length - 1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,input_feat.shape[-1])
            input_feat = torch.gather(input_feat, 1, idx).squeeze(1)
            
        if hasattr(self, 'mlp'):
            input_feat = self.mlp(input_feat)
        return input_feat
