import os
import numpy as np
import cv2
from hdf5storage import loadmat
from tqdm import tqdm
from opts import args
import pdb
import ujson as js
from collections import defaultdict

domain = args.domain
#  alias syncds="rsync -av -e ssh fating@172.18.167.17:/data/fating/HighlightDataset/proDataset/ /home/share/Highlight/proDataset/" 
def generateGT_from_youtube(path):
    count = 0
    category = []
    for root,dirs,files in os.walk(path+'/video'):
        for dr in dirs:
            category.append(dr)
    ctg_dict = defaultdict(defaultdict)
    for ctg in category:
        video_dir = os.path.join(path+'/video', ctg)
        ret = defaultdict(list)
        # with open('/home/share/Highlight/code/instagram_dataset/video_list/{}_youtube'.format(domain), 'r') as file:
        for root,dirs,files in os.walk(video_dir):
            for video in files:
                count+=1
                prefix = video.split('.')[0]
                if os.path.exists(os.path.join(path+'/label', '{}.json'.format(prefix))):
                    print(ctg+ ' processing: '+video)
                    with open(os.path.join(path+'/label', '{}.json'.format(prefix))) as label_file:
                        data = js.load(label_file)
                        flag = data[-1]
                        cap = cv2.VideoCapture(os.path.join(video_dir, video))
                        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        frames = np.zeros(int(n_frames)+1, dtype=np.int16)
                        # pdb.set_trace()
                        for idx, pair in enumerate(data[0]):
                            if flag[idx] == 1:
                                frames[int(pair[0]): int(pair[1])] = 1
                        ret[prefix] = frames[np.newaxis]
                # else:
                    # os.remove(os.path.join(root,video))
        ctg_dict[ctg] = ret
    # pdb.set_trace()
    np.save(path+'/gt_youtube.npy',ctg_dict)
    # return 

def generateGT_from_youtube_shot(path):
    count = 0
    category = []
    for root,dirs,files in os.walk(path+'/video'):
        for dr in dirs:
            category.append(dr)
    ctg_dict = defaultdict(defaultdict)
    for ctg in category:
        video_dir = os.path.join(path+'/video', ctg)
        ret = defaultdict(list)
        # with open('/home/share/Highlight/code/instagram_dataset/video_list/{}_youtube'.format(domain), 'r') as file:
        for root,dirs,files in os.walk(video_dir):
            for video in files:
                user_annot = defaultdict(list)
                count+=1
                prefix = video.split('.')[0]
                if os.path.exists(os.path.join(path+'/label', '{}.json'.format(prefix))):
                    print(ctg+ ' processing: '+video)
                    with open(os.path.join(path+'/label', '{}.json'.format(prefix))) as label_file:
                        data = js.load(label_file)
                        user_annot['shots'] = data[0]
                        new_score = []
                        for i in range(len(data[1])):
                            if data[1][i] == 1:
                                new_score.append(1)
                            else:
                                new_score.append(0)
                        print(data[1])
                        print(new_score)
                        user_annot['scores'] = new_score
                        ret[prefix].append(user_annot)
        ctg_dict[ctg] = ret
    # pdb.set_trace()
    np.save(path+'/gt_youtube.npy',ctg_dict)
    # return 


def generateGT_from_tvsum(path):
    src = os.path.join('/home/share/Highlight/orgDataset/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat')
    mat = loadmat(src)['tvsum50'][0]  # ndarray (50, )
    ret = defaultdict(defaultdict)
    idx = 1
    for element in mat:
        ctg = element[1][0][0]
        if ctg in list(ret.keys()):
            pass
        else:
            ret[ctg] = defaultdict(list)
        video_name = element[0][0][0]+'.mp4'
        print(idx,video_name)
        idx+=1
        scores = element[5].transpose().tolist() 
    
        pro_scores = []
        for sc in scores:
            sc = np.array(sc)
            
            idxs = np.argsort(-sc)
            sc[idxs[:int(len(idxs)/2)]]=1
            sc[idxs[int(len(idxs)/2):]]=0
            pro_scores.append(sc)
        scores = np.array(pro_scores)
        # pdb.set_trace()
        ret[ctg][video_name] = scores  # user_annotations, (user_nums x n_frames)
        
    np.save(path+'/gt_tvsum1.npy',ret)


def generateGT_from_tvsum1(path):
    src = os.path.join('/home/share/Highlight/orgDataset/ydata-tvsum50-v1_1/matlab/ydata-tvsum50.mat')
    mat = loadmat(src)['tvsum50'][0]  # ndarray (50, )
    ret = defaultdict(defaultdict)
    idx = 1
    for element in mat:
        ctg = element[1][0][0]
        if ctg in list(ret.keys()):
            pass
        else:
            ret[ctg] = defaultdict(list)
        video_name = element[0][0][0]+'.mp4'
        video = '/home/share/Highlight/proDataset/TVSum/video/'+ctg+'/'+video_name
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        frame_num = cap.get(7)
        print(frame_num)
        print(ctg,video_name)
        # idx+=1
        scores = element[5].transpose().tolist() 
        fps = round(fps)*2
        pro_scores = []
        for sc in scores:
            s_idxs = []
            start = -1
            user_annot = defaultdict(list)
            sc = np.array(sc)
            for i in range(len(sc)):
                if sc[i]!=start:
                    start=sc[i]
                    s_idxs.append(i)
            s_idxs.append(len(sc)-1)
            shot_idxs = []
            shot_values = []
            #划分segment
            for i in range(len(s_idxs)):
                if i==0:
                    continue
                shot_idxs.append(s_idxs[i-1])
                if s_idxs[i]-s_idxs[i-1]>fps:
                    sub_shot_num = int((s_idxs[i]-s_idxs[i-1])/fps)
                    #最后一项不要加了
                    if (s_idxs[i]-s_idxs[i-1])%fps==0:
                        sub_shot_num-=1
                    for ssn in range(sub_shot_num):
                        shot_idxs.append(s_idxs[i-1]+(ssn+1)*fps)
            shot_idxs.append(s_idxs[-1])
            for i in range(len(shot_idxs)-1):
                shot_values.append(sc[shot_idxs[i]])
            shot_values = np.array(shot_values)
            idxs = np.argsort(-shot_values)
            shot_values[idxs[:int(len(idxs)/2)]]=1
            shot_values[idxs[int(len(idxs)/2):]]=0
            user_annot['shots'] = shot_idxs
            user_annot['scores'] = shot_values
            print(len(shot_values))
            pdb.set_trace()
            ret[ctg][video_name].append(user_annot)
    np.save(path+'/gt_tvsum.npy',ret)

def generateGT_from_tvsum2(path):
    src = os.path.join('/home/share/Highlight/orgDataset/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv')
    mat = loadmat(src)  # ndarray (50, )
    pdb.set_trace()
    mat = loadmat(src)['tvsum50'][0]  # ndarray (50, )

    ret = defaultdict(defaultdict)
    idx = 1
    for element in mat:
        ctg = element[1][0][0]
        if ctg in list(ret.keys()):
            pass
        else:
            ret[ctg] = defaultdict(list)
        video_name = element[0][0][0]+'.mp4'
        print(idx,video_name)
        idx+=1
        scores = element[5].transpose().tolist() 
        pro_scores = []
        start = -1
        idxs = []
        values = []
        for sc in scores:
            sc = np.array(sc)
            for i in range(len(sc)):
                if sc[i]!=start:
                    print(sc[i])
                    start=sc[i]
                    print(value)
                    idxs.append(i)
        # pdb.set_trace()
        ret[ctg][video_name] = scores  # user_annotations, (user_nums x n_frames)
        
    np.save(path+'/gt_tvsum1.npy',ret)


def generateGT_from_CoSum(path,save):
    category = ['01_base_jump','02_bike_polo','03_eiffel_tower','04_excavators_river_xing','05_kids_playing_in_leaves','06_mlb','07_nfl','08_notre_dame_cathedral','09_statue_of_liberty','10_surfing']
    annotation_path = path+'/annotation/'
    shots_path = path+'/shots/'
    ret = defaultdict(defaultdict)
    users = ['__kk.mat','__vv.mat','__dualplus.mat']
    for ctg in category:
        ctg_ = ctg[3:]
        ret[ctg_] = defaultdict(list)
        ctg_annotation_path = annotation_path+ctg
        ctg_shorts_path = shots_path+ctg
        fileList = os.listdir(ctg_shorts_path)
        for shot in fileList:
            user_annot = defaultdict(list)
            idx = shot.split('_')[0][-1]
            shot_path = ctg_shorts_path+'/'+shot
            file = open(shot_path)
            shot_num = []
            # shot_num.append(0)
            begin=True
            for line in file.readlines():
                line=line.strip('\n')
                num = int(line)
                if begin:
                    begin=False
                    if num!=1:
                        shot_num.append(0)
                    else:
                        shot_num.append(num-1)
                else:
                    shot_num.append(num-1)

            all_shot_idx = []
            for user in users:
                annot = ctg_annotation_path+'/'+idx+user
                shot_idx = loadmat(annot)['labels'][:,0]
                all_shot_idx.append(shot_idx)
            if ctg=='10_surfing':
                pdb.set_trace()
                print('aaa')
            inter1 = list(set(all_shot_idx[0]).intersection(set(all_shot_idx[1])))
            inter2 = list(set(all_shot_idx[1]).intersection(set(all_shot_idx[2])))
            inter3 = list(set(all_shot_idx[2]).intersection(set(all_shot_idx[0])))
            gt_shot = set(inter1).union(set(inter2))
            gt_shot = list(gt_shot.union(set(inter3)))
            shot_values = []
            for i in range(len(shot_num)-1):
                if (i+1) in gt_shot:
                    shot_values.append(1)
                else:
                    shot_values.append(0)
            user_annot['shots'] = shot_num
            user_annot['scores'] = shot_values
            # pdb.set_trace()
            ret[ctg_][idx].append(user_annot)
    np.save(save+'/gt_cosum.npy',ret)
    



                

if __name__ == '__main__':
    # generateGT_from_CoSum('/home/share/Highlight/orgDataset/cosum','/home/share/Highlight/proDataset/CoSum')
    # generateGT_from_tvsum1('/home/share/Highlight/proDataset/TVSum')
    generateGT_from_youtube_shot('/home/share/Highlight/proDataset/DomainSpecific')
    # create_label
