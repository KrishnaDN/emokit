import os
import torch.nn as nn
from typing import List
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class RNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_units , bidirection=True):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, num_layers, bidirectional=bidirection, batch_first=True)
    def forward(self,x):
        '''
        x: (b,t,c)
        '''
        x,_ = self.lstm(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, 
                 num_layers=2, 
                 filters=None, 
                 kernel_sizes=None,
                 strides=None):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        assert self.num_layers==len(self.filters)==len(self.kernel_sizes)==len(self.strides)
        conv_blocks = list()
        for i in range(self.num_layers):
            if i==0:
                conv_blocks.append(nn.Conv1d(self.input_dim, self.filters[i], self.kernel_sizes[i], self.strides[i]))
                conv_blocks.append(nn.ReLU())
            else:
                conv_blocks.append(nn.Conv1d(self.filters[i-1], self.filters[i], self.kernel_sizes[i], self.strides[i]))
                conv_blocks.append(nn.ReLU())
        self.cnn = nn.Sequential(*conv_blocks)
        
    def forward(self,x):
        '''
        x: (b,t,c)
        '''
        x = self.cnn(x.transpose(1,2))
        return x.transpose(1,2)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=64, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
    
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x