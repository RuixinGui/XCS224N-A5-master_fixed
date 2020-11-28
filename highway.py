#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
class Highway(nn.Module):
    def __init__(self,word_embed_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        #used super() to call the __init__() of the nnmodule class,
        #allowing to use it in the Highway class
        super(Highway, self).__init__()
        # size of weights: word_embed_size*word_embed_size
        self.word_embed_size   = word_embed_size
        self.proj=nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate=nn.Linear(self.word_embed_size, self.word_embed_size)
    def forward(self,x_conv: torch.Tensor) -> torch.Tensor: #returns a tensor
        x_proj=F.relu(self.proj(x_conv))
        x_gate=F.sigmoid(self.gate(x_conv))
        x_highway=x_gate*x_proj+(1-x_gate)*x_conv
        return x_highway                 
### END YOUR CODE 

