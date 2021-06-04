import torch.nn as nn
from .base import BaseModel

class CrossModalAttention(BaseModel):
    def __init__(self,params):
        super(CrossModalAttention, self).__init__()
        print('CrossModalAttention')
        pass


class MultiRNNNet(BaseModel):
    def __init__(self,params):
        super(MultiRNNNet, self).__init__()
        print('MultiRNNNet')
        pass


class Wav2VecBERT(BaseModel):
    def __init__(self,params):
        super(Wav2VecBERT, self).__init__()
        print('Wav2VecBERT')
        pass
