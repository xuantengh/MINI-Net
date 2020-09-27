import os
from shutil import copy
from tqdm import tqdm
import pdb
# import ujson as js
import numpy as np
import cv2
from collections import defaultdict
from opts import args
#coding=utf8
import matplotlib.pyplot as plt
import librosa.display
import librosa
import matplotlib as mlp
import os
import shutil
import pdb
import numpy as np
from collections import defaultdict
import random
from moviepy.editor import *
from tqdm import tqdm
def mkdir_if_missed(path):
    if not os.path.exists(path):
        os.mkdir(path)


def copy_if_exist(src_file, dest_path):
    if os.path.exists(src_file):
        copy(src_file, dest_path)


def read_js(file_path):
    with open(file_path, 'r') as f:
        return js.load(f)

def save_js(data, file_path):
    with open(file_path, 'w') as f:
        f.write(js.dumps(data))

def create_mp4_time_dict(path,savepath):
    mp4_time = defaultdict(float)
    for root,dirs,files in os.walk(path):
        for fe in files:
            cap = cv2.VideoCapture(os.path.join(root,fe))    
            # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
            if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
                # get方法参数按顺序对应下表（从0开始编号)
                rate = cap.get(5)   # 帧速率
                FrameNumber = cap.get(7)  # 视频文件的帧数
                duration = FrameNumber/rate
                mp4_time[fe]=duration
                print('save:'+fe)
            else:
                print('=========error: '+fe+'==========')
    np.save(savepath,mp4_time)
def removeDim(path):
    gt_dict = np.load(path+'/gt_dict.npy').tolist()
    domains = list(gt_dict.keys())
    for domain in domains:
        gt_domain = gt_dict[domain]
        videos = list(gt_domain.keys())
        for ve in videos:
            print(path+'/feature/'+domain+'/'+ve+'.npy')
            feature = np.load(path+'/feature/'+domain+'/'+ve+'.npy').tolist()[0]
            np.save(path+'/feature/'+domain+'/'+ve+'.npy',feature)
            # pdb.set_trace()
            # print('aaa')
def clip2frame(clip_scores):
    ret = {}
    for video in tqdm(clip_scores.keys()):
        tmp = np.zeros(args.frames_per_clip*(len(clip_scores[video])+1), dtype=np.float32)
        for idx, score in enumerate(clip_scores[video]):
            tmp[args.frames_per_clip*idx: args.frames_per_clip*(idx+1)] = score
        ret[video] = tmp
    return ret
def clip2segment(clip_scores,gt):
    ret = {}
    for video in tqdm(clip_scores.keys()):
        tmp = np.zeros(args.frames_per_clip*(len(clip_scores[video])+1), dtype=np.float32)
        for idx, score in enumerate(clip_scores[video]):
            tmp[args.frames_per_clip*idx: args.frames_per_clip*(idx+1)] = score
        annoter = gt[video]
        shots = annoter[0]['shots']
        mechine = []
        for i in range(len(shots)-1):
            if shots[i+1]>len(tmp):
                break
            mechine.append(np.mean(tmp[shots[i]:shots[i+1]]))
        # mechine.append(np.mean(shots[len(shots)-1:]))
        ret[video] = mechine
    return ret
def clip2shots_youtube(clip_scores,gt):
    ret = {}
    for video in tqdm(clip_scores.keys()):
        tmp = np.zeros(args.frames_per_clip*(len(clip_scores[video])+1), dtype=np.float32)
        for idx, score in enumerate(clip_scores[video]):
            tmp[args.frames_per_clip*idx: args.frames_per_clip*(idx+1)] = score
        annoter = gt[video]
        shots = annoter[0]['shots']
        mechine = []
        for i in range(len(shots)):
            pairs = shots[i]
            if pairs[1]>len(tmp):
                break
            mechine.append(np.mean(tmp[int(pairs[0]):int(pairs[1])]))
        ret[video] = mechine
    return ret

class DealWithFeatureJson:
    def __init__(self, whole_json_path, sl_file, dest_path):
        self.whole_json_path = whole_json_path
        self.dest_path = dest_path
        with open(sl_file) as f:
            self.sl_dict = js.load(f)  # video -> long or short
        mkdir_if_missed(os.path.join(dest_path, "long_videos"))
        mkdir_if_missed(os.path.join(dest_path, "short_videos"))

    def SpiltJson(self):
        with open(self.whole_json_path, 'r') as f:
            whole_features = js.load(f)
        # pdb.set_trace()
        for item in tqdm(whole_features):
            video_name = item["video"]
            if video_name in self.sl_dict.keys():
                if self.sl_dict[video_name] == "short":
                    with open(os.path.join(self.dest_path, "short_videos", "{}.json".format(video_name)), 'w') as ff:
                        ff.write(js.dumps(item))
                if self.sl_dict[video_name] == "long":
                    with open(os.path.join(self.dest_path, "long_videos", "{}.json".format(video_name)), "w") as ff:
                        ff.write(js.dumps(item))

def __fullkey2prefix(path):
    if not 'npy' in path:
        return
    print(path)
    dicts = np.load(path, allow_pickle=True).tolist()
    # pdb.set_trace()
    keys  = list(dicts.keys())
    # new_dicts = defaultdict(list)
    # write = False
    for key in keys:
        print(key)
        if len(key.split('.'))==1:
            break
        prefix = key.split('.')[0]
        dicts[prefix] = dicts[key]
        del dicts[key]
    np.save(path,dicts)

def fullkey2prefix(path):
    for root,dirs,files in os.walk(path):
        for fe in files:
            __fullkey2prefix(os.path.join(root,re))

def wav2image(path):
        # 读取生成波形图
    x , sr = librosa.load(path)
    mlp.rcParams['axes.spines.right'] = False
    mlp.rcParams['axes.spines.top'] = False
    mlp.rcParams['axes.spines.bottom'] = False
    mlp.rcParams['axes.spines.left'] = False
    plt.figure(figsize=(14, 5))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    librosa.display.waveplot(x, sr=sr)
    plt.savefig('wav.png')

    #plt.show()
def filterAudioFeature(path):
    features = np.load(path).tolist()
    keys = list(features.keys())
    for key in keys:
        clips = features[key]
        remove = False
        for clip in clips:
            feat = clip['features']
            if not isinstance(feat,list):
                print(feat)
                remove = True
                break
        if remove:
            print('remove {}'.format(key))
            del features[key]
    np.save(path,features)

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


# 字节bytes转化kb\m\g
def recursive_get_list(path):
    result = []
    for root,dirs,files in os.walk(path):
        for dr in dirs:
            subre = recursive_get_list(os.path.join(root,dr))
            result+=subre
        for fe in files:
            if fe.split('.')[-1] in ['avi','dat','mkv','flv','vob','mp4','wmv']:
                result.append(os.path.join(root,fe)) 
    return result
# 字节bytes转化kb\m\g
def recursive_get_list_audio(path):
    result = []
    for root,dirs,files in os.walk(path):
        for dr in dirs:
            subre = recursive_get_list(os.path.join(root,dr))
            result+=subre
        for fe in files:
            if fe.split('.')[-1] =='wav':
                result.append(os.path.join(root,fe)) 
    return result
def filter_processed_video(path1,path2,set1,set2):
    set1 = [ele[len(path1):-4] for ele in set1]
    set2 = [ele[len(path2):-4] for ele in set2]
    result  = list(set(set1).difference(set(set2)))
    result = [path1+ele+'.mp4' for ele in result]
    return result
def youtube_movefile(path,savepath,labelPath):
    videos = recursive_get_list(path)
    for video in videos:
        try:
            name = video.split('/')[-1]
            path = video[:-len(name)]
            print(path+'match_label.json', labelPath+'/'+name.split('.')[0]+'.json')
            shutil.copy(path+'match_label.json', labelPath+'/'+name.split('.')[0]+'.json')
            print(video, savepath+'/'+name)
            shutil.copy(video, savepath+'/'+name)
        except Exception as e:
            print(e)
def move_to_video_without_audio(path,root):
    newPath = path[len(root):]
    video_without_audio = '/home/share/Highlight/orgDataset/video_without_audio'
    newPath = video_without_audio+newPath
    filename = newPath.split('/')[-1]
    fold = newPath[:-len(filename)]
    mkdir(fold)
    shutil.move(path,newPath)
    
def youtube_splitAudio(path,savepath):
    videos = recursive_get_list(path)
    had_audios = recursive_get_list_audio(savepath)
    # random.shuffle(videos)
    videos = filter_processed_video(path,savepath,videos,had_audios)
    print(videos)
    for video in tqdm(videos):
        
        print('================'+video)
        try:
            name = video.split('/')[-1]
            if name[0]=='.':
                # os.remove(video)
                print('delete: '+video)
                continue
            ctg = video.split('/')[-2]
            mkdir(savepath+'/'+ctg)
            if os.path.exists(savepath+'/'+ctg+'/'+name[:-3]+'wav'):
                print('pass: '+video)
                continue
            size = os.path.getsize(video)
            size = formatSize(size)
            if size<10:
                print('too small: '+video)
                # os.remove(video)
                continue
            video_obj = VideoFileClip(video)
            audio_obj = video_obj.audio
            if audio_obj is None:
                move_to_video_without_audio(video,path)
                print('No audio: '+video)
                continue
            print(video)
            audio_obj.write_audiofile(savepath+'/'+ctg+'/'+name[:-3]+'wav')
        except Exception as e:
            print(e)
            print(video)
            # os.remove(video)
def TVsum_splitAudio(path,savepath):
    videos = recursive_get_list(path)
    for video in videos:
        try:
            name = video.split('/')[-1]
            if name[0]=='.':
                # os.remove(video)
                print('delete: '+video)
                continue
            ctg = video.split('/')[-2]
            mkdir(savepath+'/'+ctg)
            if os.path.exists(savepath+'/'+ctg+'/'+name[:-3]+'wav'):
                print('pass: '+video)
                continue
            size = os.path.getsize(video)
            size = formatSize(size)
            if size<10:
                print('too small: '+video)
                # os.remove(video)
                continue
            video_obj = VideoFileClip(video)
            audio_obj = video_obj.audio
            if audio_obj is None:
                print('No audio: '+video)
                continue
            print(video)
            audio_obj.write_audiofile(savepath+'/'+ctg+'/'+name[:-3]+'wav')
        except Exception as e:
            print(e)
            # os.remove(video)

def selectUerfulVideo(domain,path):
    new_duration = defaultdict(float)
    new_video_features = defaultdict(list)
    new_audio_features = defaultdict(list)

    duration = np.load('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_duration_copy.npy',allow_pickle=True).tolist()
    video_features = np.load('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_1s_copy.npy',allow_pickle=True).tolist()
    audio_features = np.load('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_audio_edited_nopost_copy.npy',allow_pickle=True).tolist()
    names = np.load(path).tolist()
    keys_duration = list(duration.keys())
    keys_video_features = list(video_features.keys())
    keys_audio_features = list(audio_features.keys())
    counter = 0
    for name in names:
        prefix = name.split('.')[0]
        if prefix in keys_duration and prefix in keys_video_features and prefix in keys_audio_features:
            new_duration[prefix] = duration[prefix]
            new_video_features[prefix] = video_features[prefix]
            new_audio_features[prefix] = audio_features[prefix]
            print(prefix,counter)
            counter+=1
    if domain == 'surf' or domain == 'surfing':
        np.save('/home/share/Highlight/proDataset/TrainingSet/surf_duration.npy',new_duration)
        np.save('/home/share/Highlight/proDataset/TrainingSet/surf_1s.npy',new_video_features)
        np.save('/home/share/Highlight/proDataset/TrainingSet/surf_audio_edited_nopost.npy',new_audio_features)
        np.save('/home/share/Highlight/proDataset/TrainingSet/surfing_duration.npy',new_duration)
        np.save('/home/share/Highlight/proDataset/TrainingSet/surfing_1s.npy',new_video_features)
        np.save('/home/share/Highlight/proDataset/TrainingSet/surfing_audio_edited_nopost.npy',new_audio_features)
    else:
        np.save('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_duration.npy',new_duration)
        np.save('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_1s.npy',new_video_features)
        np.save('/home/share/Highlight/proDataset/TrainingSet/'+domain+'_audio_edited_nopost.npy',new_audio_features)
       
def TestUerfulVideo(domain,path):
    new_duration = defaultdict(float)
    new_video_features = defaultdict(list)
    new_audio_features = defaultdict(list)

    video_features = np.load('/home/share/Highlight/proDataset/DomainSpecific/feature/'+domain+'_1s_copy.npy',allow_pickle=True).tolist()
    audio_features = np.load('/home/share/Highlight/proDataset/DomainSpecific/feature/'+domain+'_audio_edited_nopost_copy.npy',allow_pickle=True).tolist()
    names = np.load(path).tolist()
    keys_video_features = list(video_features.keys())
    keys_audio_features = list(audio_features.keys())
    counter = 0
    for name in names:
        prefix = name.split('.')[0]
        if prefix in keys_video_features and prefix in keys_audio_features:
            new_video_features[prefix] = video_features[prefix]
            new_audio_features[prefix] = audio_features[prefix]
            print(prefix,counter)
            counter+=1
    np.save('/home/share/Highlight/proDataset/DomainSpecific/feature/'+domain+'_1s.npy',new_video_features)
    np.save('/home/share/Highlight/proDataset/DomainSpecific/feature/'+domain+'_audio_edited_nopost.npy',new_audio_features)
       
if __name__ == "__main__":
    # removeDim('/home/share/Highlight/proDataset/DomainSpecific')
    # filterAudioFeature('/home/share/Highlight/proDataset/TrainingSet/BT_1s.npy')
    # __fullkey2prefix('/home/share/Highlight/proDataset/DomainSpecific/feature/gymnastics_1s.npy')
    # __fullkey2prefix('/home/share/Highlight/proDataset/TVSum/feature/BK_1s.npy')
    # youtube_splitAudio('/home/share/Highlight/orgDataset/instagram','/home/share/Highlight/orgDataset/instagram_audio')
    # youtube_splitAudio('/home/share/Highlight/proDataset/DomainSpecific/video','/home/share/Highlight/proDataset/DomainSpecific/audio')
    # youtube_splitAudio('/home/share/Highlight/proDataset/CoSum/video','/home/share/Highlight/proDataset/CoSum/audio')

    # __fullkey2prefix('/home/share/Highlight/proDataset/TrainingSet/surfing_audio_edited_nopost.npy')

    # wav2image('/home/share/Highlight/proDataset/DomainSpecific/audio/dog/Om5mczkpcVg.wav')
    # selectUerfulVideo('surf','/home/fating/true_surf.npy')
    TestUerfulVideo('surfing','/home/fating/test_true_surfing.npy')
