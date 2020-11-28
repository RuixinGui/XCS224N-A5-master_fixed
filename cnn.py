#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,embed_size_char,num_filters, max_word_length, kernel_size=5 ):
        super(CNN,self).__init__()
        self.embed_size_char= embed_size_char # num input channels
        self.num_filters     = num_filters     # num output channels
        self.kernel_size     = kernel_size
        self.max_word_length = max_word_length
        self.conv1d=nn.Conv1d(in_channels=embed_size_char,out_channels=num_filters,
                  kernel_size=kernel_size,bias=True)
        self.max_pool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)
    def forward(self,x_reshape: torch.Tensor) -> torch.Tensor:
        # batch_size x embed_size x (m_word - kernel_size + 1)
        x_conv=self.conv1d(x_reshape)
        # maxpool and
        #Returns a tensor with all the dimensions of input of size 1 removed.
        x_conv_out=self.max_pool_1d(F.relu(x_conv)).squeeze()
        return x_conv_out
### END YOUR CODE

