from moviepy.editor import *
import pdb
from torchvggish import vggish, vggish_input
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import *
from tqdm import tqdm
import torch
def formatSize(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"

    # if kb >= 1024:
    #     M = kb / 1024
    #     if M >= 1024:
    #         G = M / 1024
    #         return  G
    #     else:
    #         return M
    # else:
    return kb
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
def get_feature(ap):
    with torch.no_grad():
        audio_name = ap.split('/')[-1]
        audio_clips_raw=[]
        audio_clips_edited=[]
        # video = VideoFileClip(os.path.join(video_path,ve))
        
        example = vggish_input.wavfile_to_examples(ap)
        if args.vgg:
            example = example.cuda()
            embeddings = embedding_model.forward(example)
        else:
            embeddings = example
        for idx,(ep,eb) in tqdm(enumerate(zip(example.cpu().numpy(),embeddings.cpu().numpy()))):
            audio_clip_raw=defaultdict(object)
            audio_clip_raw['segment']=[idx,idx+1]
            audio_clip_raw['features'] = ep
            audio_clips_raw.append(audio_clip_raw)
            if args.vgg:
                audio_clip_edited=defaultdict(object)
                audio_clip_edited['segment']=[idx,idx+1]
                audio_clip_edited['features'] = eb
                audio_clips_edited.append(audio_clip_edited)
        lock1.acquire()
        audio_dict_raw[audio_name]=audio_clips_raw
        if args.vgg:
            audio_dict_edited[audio_name]=audio_clips_edited
        lock1.release()

            
   
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--vgg', action='store_true', help='')
    

    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
if __name__=='__main__':
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock3 = threading.Lock()
    # Initialise model and download weights
    embedding_model = vggish()
    embedding_model.cuda()
    embedding_model.eval()
    args = parse_opts()
    # audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/skating')
    # audio_path_raw = '/home/share/Highlight/proDataset/TrainingSet/skating_audio_raw.npy'
    # audio_path_edited = '/home/share/Highlight/proDataset/TrainingSet/skating_audio_edited.npy'

    # audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/surfing')
    # audio_path_raw = '/home/share/Highlight/proDataset/TrainingSet/surfing_audio_raw.npy'
    # audio_path_edited = '/home/share/Highlight/proDataset/TrainingSet/surfing_audio_edited.npy'


    # audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/skiing')
    # audio_path_raw = '/home/share/Highlight/proDataset/TrainingSet/skiing_audio_raw.npy'
    # audio_path_edited = '/home/share/Highlight/proDataset/TrainingSet/skiing_audio_edited.npy'

    audio_paths = recursive_get_list('/home/share/Highlight/orgDataset/instagram_audio/dog')
    audio_path_raw = '/home/share/Highlight/proDataset/TrainingSet/dog_audio_raw.npy'
    audio_path_edited = '/home/share/Highlight/proDataset/TrainingSet/dog_audio_edited.npy'


    if os.path.exists(audio_path_raw):
        audio_dict_raw = np.load(audio_path_raw).tolist()
    else:
        audio_dict_raw =defaultdict(list)
    
    if os.path.exists(audio_path_edited):
        audio_dict_edited = np.load(audio_path_edited).tolist()
    else:
        audio_dict_edited =defaultdict(list)

    
    audio_keys = list(audio_dict_raw.keys())

    parameters = []
    passcounter = 0
    for ap in audio_paths:
        size = os.path.getsize(ap)
        size = formatSize(size)
        if size<10:
            os.remove(ap)
            continue
        audio_name = ap.split('/')[-1]
        if audio_name in audio_keys:
            print('pass: '+str(passcounter)+'  '+audio_name)
            passcounter+=1
            continue
        else:
            parameters.append(ap)
    executor = ThreadPoolExecutor(max_workers=10)
    all_task = [executor.submit(get_feature, ap) for ap in parameters]
    counter = 0
    for future in as_completed(all_task):
        counter+=1
        data = future.result()
        print("complente {}: get video [{}/{}] success".format(data,counter,len(parameters)))
