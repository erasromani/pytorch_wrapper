import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict


@gin.configurable
class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim, num_class, padding_idx=0):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        self.embed = nn.Embedding(vocab_size+2, emb_dim, padding_idx=padding_idx)
        self.linear = nn.Linear(emb_dim,num_class)
        self.config = AttrDict({
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'padding_idx': padding_idx,
        })
    
    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each document in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()     
        out = self.linear(out.float())
        return out
