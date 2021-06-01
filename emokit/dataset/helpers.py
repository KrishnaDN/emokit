import os
import numpy as np
import torch
label_dict = {'hap': 0, 'exc': 0, 'sad': 1, 'ang': 2, 'neu': 3}


def map_key2label(key):
    if len(key.split('-'))==2:
        return key.split('-')[1].split('_')[0]
    else:
        return key.split('_')[0]

def map_key2text(text_data):
    key2text = dict()
    key2label = dict()
    for item in text_data:
        key2text[item.split(' ')[0]] = ' '.join(item.split(' ')[1:]).split('___')[0]
        key2label[item.split(' ')[0]] = ' '.join(item.split(' ')[1:]).split('___')[1]
    return key2text, key2label
    