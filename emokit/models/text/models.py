import torch.nn as nn
from .base import BaseModel

class TextRNNClassifier(BaseModel):
    def __init__(self,params):
        super(TextRNNClassifier, self).__init__()
        print('TextRNNClassifier')
        pass


class TextConvNetClassifier(BaseModel):
    def __init__(self,params):
        super(TextConvNetClassifier, self).__init__()
        print('TextConvNetClassifier')
        pass
