import os
import numpy as np
import glob
import pickle
import scipy.io.wavfile as wav
import tqdm
def main(pickle_file, save_folder):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f,encoding="latin1")   
    for item in tqdm.tqdm(data):
        save_file = os.path.join(save_folder, item['id'])
        wav.write(save_file+'.wav',16000,item['signal'])
        with open(save_file+'.txt','w') as f:
            f.write(item['id']+'\t'+item['transcription'].rstrip('\r').rstrip('\n')+'___'+item['emotion'])
            f.write('\n')

if __name__=='__main__':
    save_folder ='/media/newhd/IEMOCAP_dataset/IEMOCAP_dataset'
    pickle_file = '/media/newhd/IEMOCAP_dataset/data_collected_full.pickle'
    main(pickle_file, save_folder)