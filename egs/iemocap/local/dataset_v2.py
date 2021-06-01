#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:24:15 2021

@author: krishna
"""

import os
import numpy as np
import glob
import argparse

CLASSES=['neu','ang','hap','sad','exc']

class IEMOCAPDataset:
    def __init__(self, data_folder):
        self.files = sorted(glob.glob(data_folder+'/*.wav'))
        
    
    
    def create_kaldi(self,):
        os.makedirs('data/train', exist_ok=True)
        with open('data/train/wav.scp','w') as f_wav,open('data/train/utt2spk','w') as f_u2s, open('data/train/spk2utt','w') as f_s2u, open('data/train/text','w') as f_txt:
            for filepath in self.files:
                f_wav.write(filepath.split('/')[-1]+' '+filepath+'\n')
                f_u2s.write(filepath.split('/')[-1]+' '+filepath.split('/')[-1]+'\n')
                f_s2u.write(filepath.split('/')[-1]+' '+filepath.split('/')[-1]+'\n')
                text_file = filepath[:-4]+'.txt'
                if not os.path.exists(text_file):
                    print(f'{text_file} does not exist')
                else:
                    read_text = [line.rstrip('\n') for line in open(text_file)][0]
                f_txt.write(filepath.split('/')[-1]+' '+read_text.split('\t')[1]+'\n')

            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/media/newhd/IEMOCAP_dataset/IEMOCAP_dataset')
    cmd_args = parser.parse_args()
    data_folder = cmd_args.dataset_path
    dataset = IEMOCAPDataset(data_folder)
    dataset.create_kaldi()