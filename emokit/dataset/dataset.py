import argparse
import codecs
import copy
import logging
import random
import numpy as np
import torch
import torchaudio
from transformers.file_utils import MODEL_CARD_NAME
import yaml
from PIL import Image
from PIL.Image import BICUBIC
from torch.utils.data import Dataset, DataLoader
import kaldi_io
import math
import sys
import numpy as np
from emokit.dataset.helpers import label_dict,map_key2text
from torch.utils.data import DataLoader
import warnings
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings('ignore')


class Augmentation(object):
    def __init__(self,cmvn_file):
        self.cmvn_file = cmvn_file
        self._load_kaldi_cmvn
        self.label_dict = label_dict
        
    @property
    def _load_kaldi_cmvn(self,):
        means = []
        variance = []
        with open(self.cmvn_file, 'r') as fid:
            # kaldi binary file start with '\0B'
            if fid.read(2) == '\0B':
                logging.error('kaldi cmvn binary file is not supported, please '
                            'recompute it by: compute-cmvn-stats --binary=false '
                            ' scp:feats.scp global_cmvn')
                sys.exit(1)
            fid.seek(0)
            arr = fid.read().split()
            assert (arr[0] == '[')
            assert (arr[-2] == '0')
            assert (arr[-1] == ']')
            feat_dim = int((len(arr) - 2 - 2) / 2)
            for i in range(1, feat_dim + 1):
                means.append(float(arr[i]))
            count = float(arr[feat_dim + 1])
            for i in range(feat_dim + 2, 2 * feat_dim + 2):
                variance.append(float(arr[i]))

        for i in range(len(means)):
            means[i] /= count
            variance[i] = variance[i] / count - means[i] * means[i]
            if variance[i] < 1.0e-20:
                variance[i] = 1.0e-20
            variance[i] = 1.0 / math.sqrt(variance[i])
        self.mean = means
        self.istd = variance
        

    @staticmethod
    def _spec_augmentation(x,
                        warp_for_time=False,
                        num_t_mask=2,
                        num_f_mask=2,
                        max_t=50,
                        max_f=10,
                        max_w=80):
        """ Deep copy x and do spec augmentation then return it
        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns:
            augmented feature
        """

        y = np.copy(x)
        max_frames = y.shape[0]
        max_freq = y.shape[1]

        # time warp
        if warp_for_time and max_frames > max_w * 2:
            center = random.randrange(max_w, max_frames - max_w)
            warped = random.randrange(center - max_w, center + max_w) + 1

            left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
            right = Image.fromarray(x[center:]).resize((max_freq,
                                                    max_frames - warped),
                                                    BICUBIC)
            y = np.concatenate((left, right), 0)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        return y

    @staticmethod
    def _spec_substitute(x, max_t=20, num_t_sub=3):
        """ Deep copy x and do spec substitute then return it

        Args:
            x: input feature, T * F 2D
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns:
            augmented feature
        """
        y = np.copy(x)
        max_frames = y.shape[0]
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = y[start - pos:end - pos, :]
        return y

    def _bucketing(self,):
        data=list()
        for item in self.ark_paths:
            key = item.split(' ')[0]
            ark_path = item.split(' ')[1]
            text = self.key2text[key]
            label = self.key2label[key]
            data.append((key, ark_path, text, label))
        return sorted(data, key=lambda x: len(x[2].split(' ')))

    def _load_feature(self,x):
        """ Load acoustic feature from files.
        The features have been prepared in previous step, usualy by Kaldi.
        Args:
            batch: a list of tuple (wav id , feature ark path).
        Returns:
            (keys, feats, labels)
        """
        mat = kaldi_io.read_mat(x)
        mat = mat - self.mean
        mat = mat * self.istd
        return mat


class PreProcessor(Augmentation):
    def __init__(self, 
                 data_file,
                 text_file, 
                 cmvn_file, 
                 labels,):
        super(PreProcessor, self).__init__(cmvn_file) 
        self.data_file = data_file
        self.text_file = text_file
        self.cmvn_file = cmvn_file
        self.ark_paths = [line.rstrip('\n') for line in open(self.data_file)]
        self.text_data = [line.rstrip('\n') for line in open(self.text_file)]
        self.key2text, self.key2label = map_key2text(self.text_data)
        self.data = self._bucketing()
        self.key2text, self.key2label = map_key2text(self.text_data)
        self.data = self._bucketing()
    
    def _clean_text(self, text):
        import re
        text = text.rstrip('\r').lower()
        part_clean = re.sub("[^A-Za-z0-9']+", ' ', text).split(' ')
        all_words = []
        for item in part_clean:
            if item:
                all_words.append(item)
        join_text = ' '.join(all_words)
        return join_text
    
    
    
    @property
    def _split_data(self,test_sess = 'Ses01'):
        train = list()
        test = list()
        for item in self.data:
            key, ark_path, text, label = item
            if key.split('_')[0][:2]=='sp':
                sess_name = key.split('_')[0].split('-')[1][:5]
            else:
                sess_name = key.split('_')[0][:5]
            if sess_name==test_sess:
                if key[:2]=='sp':
                    continue
                test.append((key, ark_path, text, label_dict[label]))
            else:
                train.append((key, ark_path, text, label_dict[label]))
                
        sorted_train = sorted(train, key=lambda x: len(x[2].split(' ')))
        sorted_test = sorted(test, key=lambda x: len(x[2].split(' ')))   
        return sorted_train, sorted_test



class CollateFunc(Augmentation):
    """ Collate function for AudioDataset
    """
    def __init__(self,
                 feat_dim, 
                 spec_augment=True,
                 spec_augment_conf=None,
                 spec_substitute=True,
                 spec_substitute_conf=None,
                 mode='train'):
        
        self.spec_augment = spec_augment
        self.spec_augment_conf = spec_augment_conf
        self.spec_substitute = True
        self.spec_substitute_conf = spec_substitute_conf
        self.mode = mode
        
        
    def __call__(self, batch):
        
        feats = [x[0] for x in batch]
        tokens = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        if self.mode=='train':
            if self.spec_substitute:
                spec_subs = [self._spec_substitute(x[0], **self.spec_substitute_conf) for x in batch]
                feats.extend(spec_subs)
                tokens.extend([x[1] for x in batch])
                labels.extend([x[2] for x in batch])
            if self.spec_augment:
                spec_augs = [self._spec_augmentation(x[0], **self.spec_augment_conf) for x in batch]
                feats.extend(spec_augs)
                tokens.extend([x[1] for x in batch])
                labels.extend([x[2] for x in batch])
            feats_pad = pad_sequence([torch.from_numpy(x) for x in feats], True, 0)
            text_tokens_pad = pad_sequence([torch.LongTensor(x) for x in tokens], True, 0)
            labels = torch.LongTensor([x for x in labels])
        else:
            feats_pad = pad_sequence([torch.from_numpy(x) for x in feats], True, 0)
            text_tokens_pad = pad_sequence([torch.LongTensor(x) for x in tokens], True, 0) ### Note 0 is the PAD symbol for the tokenizer
            labels = torch.LongTensor([x for x in labels])
        return feats_pad, text_tokens_pad,labels
        




class KaldiDataset(Augmentation):
    def __init__(self, 
                 data,
                 cmvn_file):
        
        super(KaldiDataset, self).__init__(cmvn_file) 
        self.data = data
        self.cmvn_file = cmvn_file
        self.tokenizer  = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, ark_path, text, label = self.data[idx]
        feats = self._load_feature(ark_path)
        text_tokens = self.tokenizer(text)['input_ids']
        return feats, text_tokens, label


if __name__=='__main__':
    config_file = '/home/krishna/Krishna/emokit/egs/iemocap/conf/transformer_v2_35.yaml'
    with open(config_file,'r') as f:
        params = yaml.safe_load(f)
    data_file = params['data']['scp_file']
    text_file = params['data']['text_file']
    labels = params['data']['labels']
    cmvn_file = params['data']['cmvn_file']
    dataset_conf = params['dataset_conf']['kaldi_offline_conf']
    processor = PreProcessor(data_file, text_file, cmvn_file, labels)
    train, test = processor._split_data
    train_dataset = KaldiDataset(train, cmvn_file,)
    collate_fun = CollateFunc(**dataset_conf, mode='train')
    train_loader = DataLoader(train_dataset,
                                   collate_fn=collate_fun,
                                   sampler=None,
                                   shuffle=False,
                                   batch_size=10)
    for i, batch in enumerate(train_loader):
        break