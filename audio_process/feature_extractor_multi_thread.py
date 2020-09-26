from moviepy.editor import *
import pdb
import librosa
import librosa.display
# from fastai.vision import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_mfcc(video_path,ve):
    
    print(ve)
    clips = video_dict[ve]
    audio_clips=[]
    video = VideoFileClip(os.path.join(video_path,ve))
    audio = video.audio
    subaudios = []
    segments = []
    for clip in clips:
        if audio is None:
            break
        # frames = int(video.duration*video.fps)
        audio_clip=defaultdict(list)
        segment = clip['segment']
        start = float(round((segment[0]-1)/video.fps,4))
        end = float(round(segment[1]/video.fps,4))
        print(start,end)
        subaudio = audio.subclip(start,end)
        subaudios.append(subaudio)
        segments.append(segment)
    try:
        for idx,(segment,subaudio) in enumerate(zip(segments,subaudios)):
            subaudio.write_audiofile(str(idx)+ve+'.mp3')
            lock2.acquire()
            y, sr = librosa.load(str(idx)+ve+'.mp3')  # 将音频文件加载为浮点时​​间系列。
            mfcc = librosa.feature.mfcc(y,sr)
            lock2.release()
            audio_clip['segment']=segment
            audio_clip['features']=mfcc
            audio_clips.append(audio_clip)
            os.remove(str(idx)+ve+'.mp3')
        lock1.acquire()
        audio_dict[ve]=audio_clips
        lock1.release()
    except Exception as e:
        print(e)
    finally:
        lock3.acquire()
        np.save(audio_path,audio_dict)
        lock3.release()
    return ve
if __name__=='__main__':
    lock1 = threading.Lock()
    lock2 = threading.Lock()
    lock3 = threading.Lock()

    # audio_path = '/home/share/Highlight/proDataset/DomainSpecific/feature/surfing_audio.npy'
    # video_feature_path = '/home/share/Highlight/proDataset/DomainSpecific/feature/surfing.npy'
    # video_path = '/home/share/Highlight/proDataset/DomainSpecific/video/surfing'

    audio_path = '/home/share/Highlight/proDataset/TrainingSet/skating_audio.npy'
    video_feature_path = '/home/share/Highlight/proDataset/TrainingSet/skating.npy'
    video_path = '/home/share/Highlight/orgDataset/instagram/skating' 

    if os.path.exists(audio_path):
        audio_dict = np.load(audio_path).tolist()
    else:
        audio_dict =defaultdict(list)
    audio_keys = list(audio_dict.keys())

    video_dict = np.load(video_feature_path).tolist()
    video_keys = list(video_dict.keys())
    parameters = []
    passcounter = 0
    for ve in video_keys:
        if ve in audio_keys:
            print('pass: '+str(passcounter)+'  '+ve)
            passcounter+=1
            continue
        else:
            parameters.append([video_path,ve])
    executor = ThreadPoolExecutor(max_workers=20)
    all_task = [executor.submit(get_mfcc, vp,ve) for [vp,ve] in parameters]

    for future in as_completed(all_task):
        data = future.result()
        print("in main: get video {}s success".format(ve))
