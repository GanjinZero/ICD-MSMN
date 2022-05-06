import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import load_embeddings


class Word_Encoder(nn.Module):
    def __init__(self, word_config={}):
        super(Word_Encoder, self).__init__()
        self.word_config = word_config
        self.word_count = self.word_config['count']
        self.word_dim = self.word_config['dim']
        self.word_padding_idx = self.word_config['padding_idx']
        self.word_dropout_prob = self.word_config['dropout']
        self.word_embedding_path = self.word_config['word_embedding_path']
        
        self.word_embedding = nn.Embedding(
            self.word_count, self.word_dim, padding_idx=self.word_padding_idx)
        if self.word_embedding_path:
            print(self.word_embedding_path)
            W = torch.Tensor(load_embeddings(self.word_embedding_path))
            assert self.word_count == W.shape[0]
            assert self.word_dim == W.shape[1]
            self.word_embedding.weight.data = W.clone()
            if self.word_config['frz']:
                self.word_embedding.weight.requires_grad = False
        self.word_dropout = nn.Dropout(self.word_dropout_prob)
        
    def forward(self, input_word):
        input_word_embed = self.word_embedding(input_word)
        input_word_embed = self.word_dropout(input_word_embed)
        return input_word_embed
