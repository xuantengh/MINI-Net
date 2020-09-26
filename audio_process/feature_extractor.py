from moviepy.editor import *
import pdb
import librosa
import librosa.display
# from fastai.vision import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
def to_mfcc(pathname):
    aud, sr = librosa.load(audio_path, sr=None)
    melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]
    # y, sr = librosa.load(pathname)  # 将音频文件加载为浮点时​​间系列。
    # mfcc = librosa.feature.mfcc(y,sr)
    pdb.set_trace()
    return melgram

def get_mfcc(video_path,clip_path,save_path):
    if os.path.exists(save_path):
        audio_dict = np.load(clip_path).tolist()
    else:
        audio_dict =defaultdict(list)
    audio_keys = list(audio_dict.keys())
    
    clips_dict = np.load(clip_path).tolist()
    video_list = list(clips_dict.keys())
    passcounter = 0
    for ve in video_list:
        if ve in audio_keys:
            print('pass: '+str(passcounter)+'  '+ve)
            passcounter+=1
            continue
        print(ve)
        clips = clips_dict[ve]
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
            # pdb.set_trace()
            subaudio = audio.subclip(start,end)
            subaudios.append(subaudio)
            segments.append(segment)
        try:
            for segment,subaudio in zip(segments,subaudios):
                subaudio.write_audiofile('temp.mp3')
                mfcc = to_mfcc('temp.mp3')
                audio_clip['segment']=segment
                audio_clip['features']=mfcc
                audio_clips.append(audio_clip)
            audio_dict[ve]=audio_clips
        except Exception as e:
            print(e)
        finally:
            np.save(save_path,audio_dict)

# get_mfcc('/home/share/Highlight/orgDataset/instagram/skating','/home/share/Highlight/proDataset/TrainingSet/skating.npy','/home/share/Highlight/proDataset/TrainingSet/skating_audio.npy')
get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/skating','/home/share/Highlight/proDataset/DomainSpecific/feature/skating.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/skating_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/surfing','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/surfing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/skiing','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/skiing_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/dog','/home/share/Highlight/proDataset/DomainSpecific/feature/dog.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/dog_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics_audio.npy')

# get_mfcc('/home/share/Highlight/proDataset/DomainSpecific/video/parkour','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour.npy','/home/share/Highlight/proDataset/DomainSpecific/feature/parkour_audio.npy')




