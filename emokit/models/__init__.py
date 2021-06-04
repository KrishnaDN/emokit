from .audio.models import ConvNetClassifier, RNNClassifier,SelfAttentionNet, TimeDelayNet, Wav2VecFintune,EmotionTransformer
from .text.models import TextConvNetClassifier, TextRNNClassifier
from .multimodal.models import CrossModalAttention, MultiRNNNet, Wav2VecBERT
AudioModels = {
    'rnn': RNNClassifier,
    'cnn': ConvNetClassifier,
    'self_attention': SelfAttentionNet,
    'tdnn': TimeDelayNet,
    'wav2vec_finetune': Wav2VecFintune,
    'emotion_transformer': EmotionTransformer
    }
    
TextModels = {
    'rnn': TextRNNClassifier,
    'cnn': TextConvNetClassifier,
    }

MultiModalModels = {
    'crossmodal_attention': CrossModalAttention,
    'multirnn': MultiRNNNet,
    'wav2vec_bert': Wav2VecBERT
    }

