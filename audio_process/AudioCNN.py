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
def get_mfcc(save_path_edited):
    audio_keys = list(audio_dict_edited.keys())
    passcounter = 0
    for ap in tqdm(audio_paths):
        try:
            torch.cuda.empty_cache()
            audio_name = ap.split('/')[-1]
            prefix = audio_name.split('.')[0]
            if prefix in audio_keys:
                print('pass: '+str(passcounter)+'  '+prefix)
                passcounter+=1
                continue
            print(args.ctg,prefix)
            # audio_clips_raw=[]
            audio_clips_edited=[]
            # video = VideoFileClip(os.path.join(video_path,ve))
            
            example = vggish_input.wavfile_to_examples(ap)
            if args.cuda:
                example = example.cuda()
            b,c,w,h = example.shape
            embeddings = embedding_model(example)
            embeddings = embeddings.view(b,-1)
            # pdb.set_trace()
            print(example.shape,embeddings.shape)
            # pdb.set_trace()
            for idx,eb in enumerate(embeddings.cpu().numpy()):
                audio_clip_edited=defaultdict(list)
                audio_clip_edited['segment']=[idx,idx+1]
                audio_clip_edited['features'] = eb.tolist()
                audio_clips_edited.append(audio_clip_edited)
            audio_dict_edited[prefix]=audio_clips_edited
        except Exception as e:
            print(e)
            np.save(save_path_edited,audio_dict_edited)

            # os.remove(ap)
    np.save(save_path_edited,audio_dict_edited)

# Initialise model and download weights
import argparse
def parse_opts():
    parser = argparse.ArgumentParser()
   
    # parser.add_argument('--ctg',type=str)
    # parser.add_argument('--dataset',type=str)
    parser.add_argument('audio_path',type=str)
    parser.add_argument('save_path',type=str)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--postprocess',action='store_true')

    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_opts()
    with torch.no_grad():
        # if args.dataset == 'instagram':
        #     audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/'+args.ctg)
        #     save_path_edited = '/home/share/Highlight/proDataset/TrainingSet/'+args.ctg+'_audio_edited'
        # if args.dataset == 'youtube':
        #     audio_paths = recursive_get_list('/home/share/Highlight/proDataset/DomainSpecific/audio/'+args.ctg)
        #     save_path_edited = '/home/share/Highlight/proDataset/DomainSpecific/feature/'+args.ctg+'_audio_edited'
        # if args.dataset == 'tvsum':
        #     audio_paths = recursive_get_list('/home/share/Highlight/proDataset/TVSum/audio/'+args.ctg)
        #     save_path_edited = '/home/share/Highlight/proDataset/TVSum/feature/'+args.ctg+'_audio_edited'
        # if args.dataset == 'cosum':
        #     audio_paths = recursive_get_list('/home/share/Highlight/proDataset/CoSum/audio/'+args.ctg)
        #     save_path_edited = '/home/share/Highlight/proDataset/CoSum/feature/'+args.ctg+'_audio_edited'

        audio_paths = args.audio_path
        save_path_edited = args.save_path

        if args.postprocess:
            save_path_edited+='.npy'
        else:
            save_path_edited +='_nopost.npy'
        embedding_model = vggish(args)
        
        if args.cuda:
            embedding_model.cuda()
        embedding_model.eval()
        #skating,skiing,surfing，dog,

        # audio_paths = recursive_get_list('/home/share/Highlight/proDataset/DomainSpecific/audio/'+args.ctg)

        print(audio_paths)
 
        if os.path.exists(save_path_edited):
            audio_dict_edited = np.load(save_path_edited,allow_pickle=True).tolist()
        else:
            audio_dict_edited =defaultdict(list)
        get_mfcc(save_path_edited)



# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/surfing','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/skiing','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/dog','/home/share/Highlight/proDataset/DomainSpecific/feature/dog.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/dog_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/parkour','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour_audio.npy')




