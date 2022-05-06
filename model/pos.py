import torch
from torch import nn
from axial_positional_embedding import AxialPositionalEmbedding


class AbsPositionEmbedding(nn.Module):
    def __init__(self, pos_config={}):
        super(AbsPositionEmbedding, self).__init__()
        self.pos_config = pos_config
        self.pos_dim = self.pos_config['rnn_dim']
        
        if self.pos_config['pos_embed'] == 'none':
            pass
        elif self.pos_config['pos_embed'] == 'learn':
            self.pos = nn.Embedding(4096, self.pos_dim) # hard code max len to 4096
            self.register_buffer("position_ids", 
                                 torch.arange(4096).expand((1, -1)))
        elif self.pos_config['pos_embed'] == 'axial':
            self.pos = AxialPositionalEmbedding(dim=self.pos_dim, axial_shape=eval(self.pos_config['axial']))
        elif self.pos_config['pos_embed'] == 'tri':
            raise NotImplementedError
        
    def forward(self, tensor_or_shape):
        if isinstance(tensor_or_shape, tuple):
            raise NotImplementedError
        
        if self.pos_config['pos_embed'] == 'none':
            return torch.zeros_like(tensor_or_shape)
        elif self.pos_config['pos_embed'] == 'learn':
            position_ids = self.position_ids[:, :tensor_or_shape.shape[1]]
            return self.pos(position_ids)
        elif self.pos_config['pos_embed'] == 'axial':
            return self.pos(tensor_or_shape)
        elif self.pos_config['pos_embed'] == 'tri':
            raise NotImplementedError
        