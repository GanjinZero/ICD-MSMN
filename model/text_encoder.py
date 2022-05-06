import torch
from torch import nn, from_numpy
import torch.nn.functional as F
import numpy as np
from model.word_encoder import Word_Encoder
from model.combiner import create_combiner


class TextEncoder(nn.Module):
    def __init__(self, word_config={}, combine_config={}):
        super(TextEncoder, self).__init__()
        self.word_config = word_config
        self.combine_config = combine_config
        
        self.input_dim = 0
        self.word_encoder = Word_Encoder(self.word_config)
        self.input_dim += self.word_config['dim']
        self.combine_config['input_dim'] = self.input_dim
        self.combiner = create_combiner(self.combine_config)
        
    def forward(self, input_word, word_mask):
        word_hidden = self.word_encoder(input_word)
        hidden = self.combiner(word_hidden, word_mask)
        return hidden
