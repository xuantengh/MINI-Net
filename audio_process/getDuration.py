import pdb
import os
# from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torchvggish import vggish, vggish_input
from tqdm import tqdm
from moviepy.editor import *

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
def get_duration(save_path_raw):
    audio_keys = list(audio_duration_dict.keys())
    passcounter = 0
    for ap in tqdm(audio_paths):
        try:
            audio_name = ap.split('/')[-1]
            if audio_name in audio_keys:
                print('pass: '+str(passcounter)+'  '+audio_name)
                passcounter+=1
                continue
            audio = AudioFileClip(ap)
            prefix = audio_name.split('.')[0]
            print(args.ctg,prefix)
            audio_duration_dict[prefix] = audio.duration
            # video = VideoFileClip(os.path.join(video_path,ve))
        except Exception as e:
            print(e)
    np.save(save_path_raw,audio_duration_dict)
# Initialise model and download weights
import argparse
def parse_opts():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--ctg',type=str)
    parser.add_argument('--dataset',type=str)
    
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_opts()
    with torch.no_grad():

       
        #skating,skiing,surfing，dog,
        if args.dataset == 'instagram':
            audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/'+args.ctg)
            duration_path = '/home/share/Highlight/proDataset/TrainingSet/'+args.ctg+'_duration.npy'
        if args.dataset == 'youtube':
        
            audio_paths = recursive_get_list('/home/share/Highlight/proDataset/DomainSpecific/audio/'+args.ctg)
            duration_path = '/home/share/Highlight/proDataset/DomainSpecific/feature/'+args.ctg+'_duration.npy'
        if args.dataset == 'tvsum':
            audio_paths = recursive_get_list('/home/share/Highlight/proDataset/TVSum/audio/'+args.ctg)
            duration_path = '/home/share/Highlight/proDataset/TVSum/feature/'+args.ctg+'_duration.npy'
        if os.path.exists(duration_path):
            audio_duration_dict = np.load(duration_path).tolist()
        else:
            audio_duration_dict =defaultdict(float)
        print(duration_path)
    
        get_duration(duration_path)



# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/surfing','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/skiing','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/dog','/home/share/Highlight/proDataset/DomainSpecific/feature/dog.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/dog_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/parkour','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour_audio.npy')




