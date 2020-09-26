import pdb
import os
# from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torchvggish import vggish, vggish_input
from tqdm import tqdm
import torch
def recursive_get_list(path):
    result = []
    for root,dirs,files in os.walk(path):
        for dr in dirs:
            subre = recursive_get_list(os.path.join(root,dr))
            result+=subre
        for fe in files:
            # if fe.split('.')[-1] in ['avi','dat','mkv','flv','vob','mp4','wmv']:
            result.append(os.path.join(root,fe)) 
    return result
def get_mfcc(save_path_raw):
    audio_keys = list(audio_dict_raw.keys())
    passcounter = 0
    try:
        for ap in tqdm(audio_paths):
            audio_name = ap.split('/')[-1]
            if audio_name in audio_keys:
                print('pass: '+str(passcounter)+'  '+audio_name)
                passcounter+=1
                continue
            print(args.ctg,audio_name)
            audio_clips_raw=[]
            # video = VideoFileClip(os.path.join(video_path,ve))
            
            example = vggish_input.wavfile_to_examples(ap,return_tensor=False)
            print(example.shape)

            for idx,(ep) in enumerate(zip(example)):
                audio_clip_raw=defaultdict(list)
                audio_clip_raw['segment']=[idx,idx+1]
                audio_clip_raw['features'] = ep
                audio_clips_raw.append(audio_clip_raw)
            audio_dict_raw[audio_name]=audio_clips_raw
    except Exception as e:
        print(e)
    finally:
        np.save(save_path_raw,audio_dict_raw)


# Initialise model and download weights
import argparse
def parse_opts():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--ctg',type=str)

    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_opts()
    with torch.no_grad():

        audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/'+args.ctg)
        print(audio_paths)
        save_path_raw = '/home/share/Highlight/proDataset/TrainingSet/'+args.ctg+'_audio_raw.npy'
        if os.path.exists(save_path_raw):
            audio_dict_raw = np.load(save_path_raw).tolist()
        else:
            audio_dict_raw =defaultdict(list)
        get_mfcc(save_path_raw)

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/surfing','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/skiing','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/dog','/home/share/Highlight/proDataset/DomainSpecific/feature/dog.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/dog_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/parkour','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour_audio.npy')




