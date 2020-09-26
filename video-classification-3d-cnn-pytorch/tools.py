import os
import shutil
import pdb
import numpy as np
from collections import defaultdict
from moviepy.editor import *
def merge(path,save):
    outputs = defaultdict(list)
    # save = path.split('/')[-1]
    for root,dirs,files in os.walk(path):
        for fe in files:
            print(fe[:-4])
            p = os.path.join(root,fe)
            features = np.load(p,allow_pickle=True).tolist()
            outputs[fe[:-4]] = features
    np.save(save,outputs)

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
def youtube_splitAudio(path,savepath):
    videos = recursive_get_list(path)
    for video in videos:
        try:
            name = video.split('/')[-1]
            if name[0]=='.':
                os.remove(video)
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
                os.remove(video)
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
            os.remove(video)
def TVsum_splitAudio(path,savepath):
    videos = recursive_get_list(path)
    for video in videos:
        try:
            name = video.split('/')[-1]
            if name[0]=='.':
                os.remove(video)
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
                os.remove(video)
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
            os.remove(video)
       

def tvsum_movefile(path,orgPath):
    gt_dict = np.load(path+'/gt_tvsum.npy').tolist()
    ctgs = list(gt_dict.keys())
    idx = 1
    for ctg in ctgs:
        mkdir(path+'/video/'+ctg)
        files = list(gt_dict[ctg].keys())
        for fe in files:
            print(idx, fe)
            idx+=1
            shutil.copy(orgPath+'/video/'+fe,path+'/video/'+ctg+'/'+fe)
    shutil.copy(orgPath+'/matlab/ydata-tvsum50.mat',path+'/label/ydata-tvsum50.mat')
        
# TVsum_splitAudio('/home/share/Highlight/proDataset/TVSum/video','/home/share/Highlight/proDataset/TVSum/audio')
youtube_splitAudio('/home/share/Highlight/orgDataset/instagram','/home/share/Highlight/orgDataset/instagram_audio')
# tvsum_movefile('/home/share/Highlight/proDataset/TVSum','/home/share/Highlight/orgDataset/ydata-tvsum50-v1_1')

# movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/dog','/home/share/Highlight/proDataset/DomainSpecific/video/dog','/home/share/Highlight/proDataset/DomainSpecific/label')
# movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/video/gymnastics','/home/share/Highlight/proDataset/DomainSpecific/label')
# movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/parkour','/home/share/Highlight/proDataset/DomainSpecific/video/parkour','/home/share/Highlight/proDataset/DomainSpecific/label')
# movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/skating','/home/share/Highlight/proDataset/DomainSpecific/video/skating','/home/share/Highlight/proDataset/DomainSpecific/label')
# youtube_movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/skiing','/home/share/Highlight/proDataset/DomainSpecific/video/skiing','/home/share/Highlight/proDataset/DomainSpecific/label')
# youtube_movefile('/home/share/Highlight/orgDataset/DomainSpecificHighlight/surfing','/home/share/Highlight/proDataset/DomainSpecific/video/surfing','/home/share/Highlight/proDataset/DomainSpecific/label')





# merge('/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/skating','/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/skating.npy')
# merge('/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/dog','/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/dog.npy')
# merge('/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/parkour','/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/parkour.npy')
# merge('/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/gymnstics','/data/fating/HighlightDataset/proDataset/DomainSpecific/feature/gymnstics.npy')

# getlist('/home/share/Highlight/orgDataset/instagram/skating','video_list.txt')
