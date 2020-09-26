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
def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
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
        print(audio_name)
        audio_clips_raw=[]
        audio_clips_edited=[]
        # video = VideoFileClip(os.path.join(video_path,ve))
        
        example = vggish_input.wavfile_to_examples(ap)
        embeddings = embedding_model.forward(example)
        print(example.shape,embeddings.shape)

        for idx,(ep,eb) in tqdm(enumerate(zip(example.cpu().numpy(),embeddings.cpu().numpy()))):
            audio_clip_raw=defaultdict(object)
            audio_clip_raw['segment']=[idx,idx+1]
            audio_clip_raw['features'] = ep
            audio_clips_raw.append(audio_clip_raw)
            audio_clip_edited=defaultdict(object)
            audio_clip_edited['segment']=[idx,idx+1]
            audio_clip_edited['features'] = eb
            audio_clips_edited.append(audio_clip_edited)
        lock1.acquire()
        audio_dict_raw[audio_name]=audio_clips_raw
        audio_dict_edited[audio_name]=audio_clips_edited
        lock1.release()

        lock2.acquire()
        np.save(audio_path_raw,audio_dict_raw)
        np.save(audio_path_edited,audio_dict_edited)
        lock2.release()
        return ap
def ctg_process(audio_path,audio_path_raw,audio_path_edited):
    
    # Initialise model and download weights
    print(audio_path)
    print(audio_path_raw)
    print(audio_path_edited)
    audio_paths = recursive_get_list(audio_path)

    
    audio_keys = list(audio_dict_raw.keys())

    parameters = []
    passcounter = 0
    for ap in audio_paths:
        audio_name = ap.split('/')[-1]
        if audio_name in audio_keys:
            print('pass: '+str(passcounter)+'  '+audio_name)
            passcounter+=1
            continue
        else:
            parameters.append(ap)
    executor = ThreadPoolExecutor(max_workers=3)
    all_task = [executor.submit(get_feature, ap) for ap in parameters]

    for future in as_completed(all_task):
        data = future.result()
        print("in main: get video {}s success".format(data))

if __name__=='__main__':
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock3 = threading.Lock()
    with torch.no_grad():
        embedding_model = vggish()
        embedding_model.eval()
        audio_folder = '/home/share/Highlight/orgDataset/instagram_audio/'
        save_path = '/home/share/Highlight/proDataset/TrainingSet/'
        folders = []
        ctg = []
        for root,dirs,files in os.walk(audio_folder):
            for dr in dirs:
                folders.append(os.path.join(root,dr))
                ctg.append(dr)
        for audio_path, ctg in zip(folders,ctg):
            audio_path_raw = save_path+'/'+ctg+'_audio_raw.npy'
            audio_path_edited = save_path+'/'+ctg+'_audio_edited.npy'
            if os.path.exists(audio_path_raw):
                audio_dict_raw = np.load(audio_path_raw).tolist()
            else:
                audio_dict_raw =defaultdict(list)

            if os.path.exists(audio_path_edited):
                audio_dict_edited = np.load(audio_path_edited).tolist()
            else:
                audio_dict_edited =defaultdict(list)
            ctg_process(audio_path,audio_path_raw,audio_path_edited)
