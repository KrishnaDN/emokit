import torch.nn as nn
import torch
from .base import BaseModel
from typing import List, Optional 
from .blocks import CNN, RNN, MultiHeadSelfAttention, Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SelfAttentionNet(BaseModel):
    def __init__(self,config):
        super(SelfAttentionNet, self).__init__()
        self.cnn_conf = config['cnn_conf']
        self.rnn_conf = config['rnn_conf']
        self.attn_conf = config['attention_conf']
        self.num_classes = config['num_classes']
        self.input_size = config['input_size']
        self.cnn = CNN(self.input_size, **self.cnn_conf)
        self.rnn = RNN(self.cnn_conf['filters'][-1], **self.rnn_conf)
        self.attention = MultiHeadSelfAttention(dim=self.rnn_conf['hidden_units'] * (int(self.rnn_conf['bidirection'])+1), **self.attn_conf)
        self.pool = nn.AdaptiveAvgPool1d((1))
        self.classifier = nn.Linear(self.rnn_conf['hidden_units'] * (int(self.rnn_conf['bidirection'])+1), self.num_classes)
        self.crit = nn.CrossEntropyLoss()
        
    def forward(self, inputs, targets):
        out = self.cnn(inputs)
        out = self.rnn(out)
        out = self.attention(out)
        probs = self.classifier(self.pool(out.transpose(1,2)).squeeze(2))
        loss = self.compute_loss(probs, targets)
        return loss, torch.topk(probs,1)[1].squeeze(1)
        
    def compute_loss(self, probs, target):
        loss = self.crit(probs, target)
        return loss

    def inference(self, inputs):
        out = self.cnn(inputs)
        out = self.rnn(out)
        out = self.attention(out)
        probs = self.classifier(self.pool(out.transpose(1,2)).squeeze(2))
        return torch.topk(probs,1)[1].squeeze(1)


class EmotionTransformer(BaseModel):
    def __init__(self,config):
        super(EmotionTransformer, self).__init__()
        input_size = config['input_size']
        patch_size = config['patch_size']
        channels = config['channels']
        depth = config['depth']
        heads = config['heads']
        pool = config['pool']
        dim = config['dim']
        emb_dropout = config['emb_dropout']
        mlp_dim = config['mlp_dim']
        num_classes = config['num_classes']
        dim_head = config['dim_head']
        
        num_patches = int(input_size[0]/patch_size[0] * input_size[1]/patch_size[1])
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=0.2)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.crit = nn.CrossEntropyLoss()
        
    
    def forward(self, inputs, target):
        inputs = inputs.unsqueeze(1)
        x = self.to_patch_embedding(inputs)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        probs = self.mlp_head(x)
        loss = self.compute_loss(probs, target)
        return loss, torch.topk(probs,1)[1].squeeze(1)
        
    def compute_loss(self, probs, target):
        loss = self.crit(probs, target)
        return loss

    def inference(self, inputs):
        x = self.to_patch_embedding(inputs)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        probs = self.mlp_head(x)
        return torch.topk(probs,1)[1].squeeze(1)



class ConvNetClassifier(BaseModel):
    def __init__(self,params):
        super(ConvNetClassifier, self).__init__()
        print('ConvNetClassifier')
        pass



class RNNClassifier(BaseModel):
    def __init__(self,params):
        super(RNNClassifier, self).__init__()
        print('RNNClassifier')
        pass



class TimeDelayNet(BaseModel):
    def __init__(self,params):
        super(TimeDelayNet, self).__init__()
        print('TimeDelayNet')
        pass
    
    
class Wav2VecFintune(BaseModel):
    def __init__(self,params):
        super(Wav2VecFintune, self).__init__()
        print('Wav2VecFintune')
        pass
    
    
