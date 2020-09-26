import torch
from torch.utils.data import Dataset
# import ujson as js
import random
import pdb
import os
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from opts import args
import pdb
import random
from tqdm import tqdm 
import copy
class TemporalPairs(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        self.short_videos = []
        self.long_videos = []
        #2128327541805730762 11
        if args.paper_data_split:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper and self.duration[key]>args.short_lower: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower and self.duration[key]<args.long_upper:
                    self.long_videos.append(key)
        else:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower:
                    self.long_videos.append(key)
        self.short_dict = defaultdict(list)
        # pdb.set_trace()
        short_idx = 0
        for prefix in tqdm(self.short_videos):
            # prefix = short_video.split('.')[0]

            if prefix not in video_keys:
                continue
            if prefix not in audio_keys:
                continue
            
            afeatures = self.audio_features[prefix]
            vfeatures = self.video_features[prefix]

            feat_len = len(afeatures) if len(afeatures)<len(vfeatures) else len(vfeatures)
            for i in range(feat_len):
                self.short_dict[short_idx] = [prefix,i]
                short_idx+=1

        self.long_dict = defaultdict(list)
        long_idx = 0
        for prefix in tqdm(self.long_videos):
            # features = np.load(long_video,allow_pickle=True ).tolist()
            # prefix = long_video.split('.')[0]
            if prefix not in video_keys:
                continue
            if prefix not in audio_keys:
                continue
            afeatures = self.audio_features[prefix]
            vfeatures = self.video_features[prefix]
            feat_len = len(afeatures) if len(afeatures)<len(vfeatures) else len(vfeatures)
            for i in range(feat_len):
                self.long_dict[long_idx] = [prefix,i]
                long_idx+=1

        self.GeneratePairs() # n x num_per_group x 2
    def GeneratePairs(self):
        # pdb.set_trace()
        groups = []
        pairs = []
        short_keys = list(self.short_dict.keys())
        long_keys = list(self.long_dict.keys())
        if len(long_keys)>len(short_keys):
            #make them the same length
            re = int(len(long_keys)/len(short_keys))
            remainder = len(long_keys)%len(short_keys)
            new_short_keys = []
            for i in range(re):
                new_short_keys+=short_keys
            new_short_keys += random.sample(short_keys, remainder)
            short_keys = new_short_keys

            #shuffle
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            #construct pair
            pairs = list(zip(short_keys,long_keys))

        else:
            re = int(len(short_keys)/len(long_keys))
            remainder = len(short_keys)%len(long_keys)
            new_lone_keys = []
            for i in range(re):
                new_lone_keys+=long_keys
            new_lone_keys += random.sample(long_keys, remainder)
            long_keys = new_lone_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys,long_keys))
        # pdb.set_trace()
        total_pairs = len(pairs)
        remainder = total_pairs % self.num_per_group
        left=random.sample(pairs, self.num_per_group-remainder)
        pairs+=left
        self.groups = np.array(pairs).reshape(-1,args.num_per_group,2).tolist()
        return

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        
        vXI = torch.Tensor()
        vXJ = torch.Tensor()
        aXI = torch.Tensor()
        aXJ = torch.Tensor()
        group = self.groups[idx] # num_per_group x 2
        for pair in group:
            (s_key, l_key) = pair
            [prefix,idx] = self.short_dict[s_key]
            length = len(self.video_features[prefix])
            vxi = []
            if idx<int(args.temporalN/2):
                vxi+=([[0]*args.feature_dim for i in range(int(args.temporalN/2)-idx)])
                vxi+=([self.video_features[prefix][i]['features'] for i in range(idx+1)])
            else:
                vxi+=([self.video_features[prefix][i]['features'] for i in range(idx-int(args.temporalN/2),idx+1)])

            if idx>=length-int(args.temporalN/2):
                vxi+=([self.video_features[prefix][i]['features'] for i in range(idx+1,length)])
                vxi+=([[0]*args.feature_dim for i in range(idx - (length-int(args.temporalN/2))+1)])
            else:
                vxi+=([self.video_features[prefix][i]['features'] for i in range(idx+1,idx+int(args.temporalN/2)+1)])

            # audio = video.split('.')[0]+'.wav'
            length = len(self.audio_features[prefix])
            axi = []
            if idx<int(args.temporalN/2):
                axi+=([[0]*args.audio_dim for i in range(int(args.temporalN/2)-idx)])
                axi+=([self.audio_features[prefix][i]['features'] for i in range(idx+1)])
            else:
                axi+=([self.audio_features[prefix][i]['features'] for i in range(idx-int(args.temporalN/2),idx+1)])

            if idx>=length-int(args.temporalN/2):
                axi+=([self.audio_features[prefix][i]['features'] for i in range(idx+1,length)])
                axi+=([[0]*args.audio_dim for i in range(idx - (length-int(args.temporalN/2))+1)])
            else:
                # print(idx,length,int(args.temporalN/2))
                axi+=([self.audio_features[prefix][i]['features'] for i in range(idx+1,idx+int(args.temporalN/2)+1)])



            [prefix,idx] = self.long_dict[l_key]

            length = len(self.video_features[prefix])
            vxj = []
            if idx<int(args.temporalN/2):
                vxj+=([[0]*args.feature_dim for i in range(int(args.temporalN/2)-idx)])
                vxj+=([self.video_features[prefix][i]['features'] for i in range(idx+1)])
            else:
                vxj+=([self.video_features[prefix][i]['features'] for i in range(idx-int(args.temporalN/2),idx+1)])

            if idx>=length-int(args.temporalN/2):
                vxj+=([self.video_features[prefix][i]['features'] for i in range(idx+1,length)])
                vxj+=([[0]*args.feature_dim for i in range(idx - (length-int(args.temporalN/2))+1)])
            else:
                vxj+=([self.video_features[prefix][i]['features'] for i in range(idx+1,idx+int(args.temporalN/2)+1)])
            # audio = video.split('.')[0]+'.wav'
            length = len(self.audio_features[prefix])
            axj = []
            if idx<int(args.temporalN/2):
                axj+=([[0]*args.audio_dim for i in range(int(args.temporalN/2)-idx)])
                axj+=([self.audio_features[prefix][i]['features'] for i in range(idx+1)])
            else:
                axj+=([self.audio_features[prefix][i]['features'] for i in range(idx-int(args.temporalN/2),idx+1)])

            if idx>=length-int(args.temporalN/2):
                axj+=([self.audio_features[prefix][i]['features'] for i in range(idx+1,length)])
                axj+=([[0]*args.audio_dim for i in range(idx - (length-int(args.temporalN/2))+1)])
            else:
                axj+=([self.audio_features[prefix][i]['features'] for i in range(idx+1,idx+int(args.temporalN/2)+1)])
            # print(len(vxi))

            vxi = torch.Tensor(vxi)
            vxj = torch.Tensor(vxj)
            vxi = vxi.unsqueeze(0)
            vxj = vxj.unsqueeze(0)

            vXI = torch.cat((vXI, vxi), dim=0)
            vXJ = torch.cat((vXJ, vxj), dim=0)
            axi = torch.Tensor(axi)
            axj = torch.Tensor(axj)
            axi = axi.unsqueeze(0)
            axj = axj.unsqueeze(0)
            aXI = torch.cat((aXI, axi), dim=0)
            aXJ = torch.cat((aXJ, axj), dim=0)

        return vXI, vXJ,aXI,aXJ


class Pairs(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        self.short_videos = []
        self.long_videos = []
        #2128327541805730762 11
        # pdb.set_trace()
        if args.paper_data_split:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper and self.duration[key]>args.short_lower: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower and self.duration[key]<args.long_upper:
                    self.long_videos.append(key)
        else:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower:
                    self.long_videos.append(key)
        self.short_dict = defaultdict(list)
        # pdb.set_trace()
        short_idx = 0
        for prefix in tqdm(self.short_videos):
            # prefix = short_video.split('.')[0]

            if prefix not in video_keys:
                continue
            if prefix not in audio_keys:
                continue
            
            afeatures = self.audio_features[prefix]
            vfeatures = self.video_features[prefix]

            feat_len = len(afeatures) if len(afeatures)<len(vfeatures) else len(vfeatures)
            for i in range(feat_len):
                self.short_dict[short_idx] = [prefix,i]
                short_idx+=1

        self.long_dict = defaultdict(list)
        long_idx = 0
        for prefix in tqdm(self.long_videos):
            # features = np.load(long_video,allow_pickle=True ).tolist()
            # prefix = long_video.split('.')[0]
            if prefix not in video_keys:
                continue
            if prefix not in audio_keys:
                continue
            afeatures = self.audio_features[prefix]
            vfeatures = self.video_features[prefix]
            feat_len = len(afeatures) if len(afeatures)<len(vfeatures) else len(vfeatures)
            for i in range(feat_len):
                self.long_dict[long_idx] = [prefix,i]
                long_idx+=1

        self.GeneratePairs() # n x num_per_group x 2
    def GeneratePairs(self):
        # pdb.set_trace()
        groups = []
        pairs = []
        short_keys = list(self.short_dict.keys())
        long_keys = list(self.long_dict.keys())
        if len(long_keys)>len(short_keys):
            #make them the same length
            re = int(len(long_keys)/len(short_keys))
            remainder = len(long_keys)%len(short_keys)
            new_short_keys = []
            for i in range(re):
                new_short_keys+=short_keys
            new_short_keys += random.sample(short_keys, remainder)
            short_keys = new_short_keys

            #shuffle
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            #construct pair
            pairs = list(zip(short_keys,long_keys))

        else:
            re = int(len(short_keys)/len(long_keys))
            remainder = len(short_keys)%len(long_keys)
            new_lone_keys = []
            for i in range(re):
                new_lone_keys+=long_keys
            new_lone_keys += random.sample(long_keys, remainder)
            long_keys = new_lone_keys
            random.shuffle(long_keys)
            random.shuffle(short_keys)
            pairs = list(zip(short_keys,long_keys))
        # pdb.set_trace()
        total_pairs = len(pairs)
        remainder = total_pairs % self.num_per_group
        left=random.sample(pairs, self.num_per_group-remainder)
        pairs+=left
        self.groups = np.array(pairs).reshape(-1,args.num_per_group,2).tolist()
        return

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        
        vXI = torch.Tensor()
        vXJ = torch.Tensor()
        aXI = torch.Tensor()
        aXJ = torch.Tensor()
        group = self.groups[idx] # num_per_group x 2
        for pair in group:
            (s_key, l_key) = pair
            [prefix,idx] = self.short_dict[s_key]
            vxi =  self.video_features[prefix][idx]['features']
            # audio = video.split('.')[0]+'.wav'
            axi = self.audio_features[prefix][idx]['features']
            [prefix,idx] = self.long_dict[l_key]
       
            vxj = self.video_features[prefix][idx]['features']
            # audio = video.split('.')[0]+'.wav'
            axj = self.audio_features[prefix][idx]['features']
            
            vxi = torch.Tensor(vxi).view(1, -1)
            vxj = torch.Tensor(vxj).view(1, -1)
            vXI = torch.cat((vXI, vxi), dim=0)
            vXJ = torch.cat((vXJ, vxj), dim=0)
            axi = torch.Tensor(axi).view(1, -1)
            axj = torch.Tensor(axj).view(1, -1)
            aXI = torch.cat((aXI, axi), dim=0)
            aXJ = torch.cat((aXJ, axj), dim=0)

        return vXI, vXJ,aXI,aXJ

class SamePairs(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        # keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        # self.short_videos = []
        # self.long_videos = []
        self.keys = list(set(video_keys).intersection(set(audio_keys)))
        for key in self.keys:
            videos = self.video_features[key]
            audios = self.video_features[key]
            if len(videos)<2 or len(audios)<2:
                self.keys.remove(key)
    def GeneratePairs(self):
       
        return

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        videos = self.video_features[key]
        audios = self.audio_features[key]
        length = len(videos) if len(videos)<len(audios) else len(audios)
        # print(length)
        idxs = list(range(length))
        select_idx = random.sample(idxs,2)
        vxi = torch.tensor(videos[select_idx[0]]['features'])
        vxj = torch.tensor(videos[select_idx[1]]['features'])

        axi = torch.tensor(audios[select_idx[0]]['features'])
        axj = torch.tensor(audios[select_idx[1]]['features'])
        return vxi, vxj,axi,axj

class MILDataset(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        if args.dataset == 'youtube':
            self.domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
        if args.dataset == 'tvsum':
            self.domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
        if args.dataset == 'cosum':
            self.domains = ['base_jump', 'bike_polo', 'eiffel_tower', 'excavators_river_cross', 'kids_playing_in_leaves', 
                            'MLB', 'NFL', 'notre_dame_cathedral', 'statue_of_liberty',  'surf']
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        self.domains.remove(domain)
        self.negBags_video = defaultdict(list)
        self.negBags_audio = defaultdict(list)
        self.negBags_duration = defaultdict(float)
        for bag in self.domains:
            duration = np.load(train_path+'/'+bag+'_duration.npy',allow_pickle=True).tolist()
            video_features = np.load(train_path+'/'+bag+'_1s.npy',allow_pickle=True).tolist()
            # pdb.set_trace()
            audio_features = np.load(train_path+'/'+bag+'_audio_edited_nopost.npy',allow_pickle=True).tolist()
            self.negBags_video = {**self.negBags_video,**video_features}
            self.negBags_audio = {**self.negBags_audio,**audio_features}
            self.negBags_duration = {**self.negBags_duration,**duration}
        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        commonKeys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 
        self.posKeys = []
        for key in tqdm(commonKeys):
            if self.duration[key]<=args.short_upper and self.duration[key]>=args.short_lower:
                self.posKeys.append(key)
        duration_keys = []
        for key in list(self.negBags_duration.keys()):
            if self.negBags_duration[key]>=args.long_lower:
                duration_keys.append(key)
        video_keys = list(self.negBags_video.keys())
        audio_keys = list(self.negBags_audio.keys())
        self.negKeys = list(set(duration_keys).intersection(set(video_keys).intersection(set(audio_keys))))
        self.availableKeys = copy.deepcopy(self.posKeys)
        
        if len(self.negKeys) > len(self.posKeys):
            re = int(len(self.negKeys)/len(self.posKeys))
            remainder = len(self.negKeys)%len(self.posKeys)
            new_posKeys = []
            for i in range(re):
                new_posKeys+=self.posKeys
            new_posKeys += random.sample(self.posKeys, remainder)
            self.posKeys = new_posKeys
        else:
            re = int(len(self.posKeys)/len(self.negKeys))
            remainder = len(self.posKeys)%len(self.negKeys)
            new_negKeys = []
            for i in range(re):
                new_negKeys+=self.negKeys
            new_negKeys += random.sample(self.negKeys, remainder)
            self.negKeys = new_negKeys
        self.GeneratePairs()
    def GeneratePairs(self):
        #shuffle
        random.shuffle(self.posKeys)
        random.shuffle(self.negKeys)
        self.pairs = list(zip(self.posKeys,self.negKeys))
        return

    def __len__(self):
        return len(self.negKeys) if len(self.negKeys) > len(self.availableKeys) else len(self.availableKeys)

    def __getitem__(self, idx):
        (posKey,negKey) = self.pairs[idx]
        video_seg = self.video_features[posKey]
        audio_seg = self.audio_features[posKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat = random.sample(vfeat,args.bagsize)
            afeat = random.sample(afeat,args.bagsize)
        pos_vfeat = torch.tensor(vfeat)
        pos_afeat = torch.tensor(afeat)

        video_seg = self.negBags_video[negKey]
        audio_seg = self.negBags_audio[negKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat = random.sample(vfeat,args.bagsize)
            afeat = random.sample(afeat,args.bagsize)
        neg_vfeat = torch.tensor(vfeat)
        neg_afeat = torch.tensor(afeat)
        pos_label = torch.ones(1,)
        neg_label = torch.zeros(1,)


        return pos_vfeat,pos_afeat,neg_vfeat,neg_afeat,pos_label,neg_label

class MTMILDataset(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        if args.dataset == 'youtube':
            self.domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
        if args.dataset == 'tvsum':
            self.domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
        if args.dataset == 'cosum':
            self.domains = ['statue_of_liberty','eiffel_tower','NFL','kids_playing_in_leaves','MLB','excavators_river_cross','notre_dame_cathedral',
                            'surf','bike_polo','base_jump']
        
        self.short_form=[]
        self.long_form=[]
       
        self.vfeats = defaultdict(list)
        self.afeats = defaultdict(list)
        for bag in self.domains:
            domain_short = []
            domain_long = []
            duration = np.load(train_path+'/'+bag+'_duration.npy',allow_pickle=True).tolist()
            video_features = np.load(train_path+'/'+bag+'_1s.npy',allow_pickle=True).tolist()
            audio_features = np.load(train_path+'/'+bag+'_audio_edited_nopost.npy',allow_pickle=True).tolist()
            self.vfeats = {**self.vfeats,**video_features}
            self.afeats = {**self.afeats,**audio_features}

            keys = list(duration.keys())
            video_keys = list(video_features.keys())
            audio_keys = list(audio_features.keys())
            commonKeys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 
            for key in tqdm(commonKeys):
                if duration[key]<=args.short_upper and duration[key]>=args.short_lower:
                    domain_short.append(key)
                if duration[key]>=args.long_lower:
                    domain_long.append(key)
            self.short_form.append(domain_short)
            self.long_form.append(domain_long)
        self.negBags = []
        for i in range(len(self.long_form)):
            domain_negBag = []
            for j in range(len(self.long_form)):
                if j==i:
                    continue
                domain_negBag +=self.long_form[j]
            self.negBags.append(domain_negBag)
    def __len__(self):
        max_ = 0
        for domain_short in self.short_form:
            if len(domain_short)>max_:
                max_ = len(domain_short)
        return max_

    def __getitem__(self, idx):
        pos_vfeats,pos_afeats,neg_vfeats,neg_afeats,pos_labels,neg_labels=[],[],[],[],[],[]
        for i in range(len(self.short_form)):
            real_idx = idx%len(self.short_form[i])
            posKey = self.short_form[i][real_idx]
            random.shuffle(self.negBags[i])
            negKey = random.sample(self.negBags[i],1)[0]

            video_seg = self.vfeats[posKey]
            audio_seg = self.afeats[posKey]
            vfeat = []
            afeat = []
            for vseg,aseg in zip(video_seg,audio_seg):
                vfeat.append(vseg['features'])
                afeat.append(aseg['features'])
            if len(vfeat) <= args.bagsize:
                re = int(args.bagsize/len(vfeat))
                remainder = args.bagsize%len(vfeat)
                new_vfeat = []
                new_afeat = []
                for k in range(re):
                    new_vfeat+=vfeat
                    new_afeat+=afeat
                new_vfeat += vfeat[:remainder]
                new_afeat += afeat[:remainder]
                vfeat = new_vfeat
                afeat = new_afeat
            else:
                vfeat = random.sample(vfeat,args.bagsize)
                afeat = random.sample(afeat,args.bagsize)
            pos_vfeat = torch.tensor(vfeat)
            pos_afeat = torch.tensor(afeat)

            video_seg = self.vfeats[negKey]
            audio_seg = self.afeats[negKey]
            vfeat = []
            afeat = []
            for vseg,aseg in zip(video_seg,audio_seg):
                vfeat.append(vseg['features'])
                afeat.append(aseg['features'])
            if len(vfeat) <= args.bagsize:
                re = int(args.bagsize/len(vfeat))
                remainder = args.bagsize%len(vfeat)
                new_vfeat = []
                new_afeat = []
                for k in range(re):
                    new_vfeat+=vfeat
                    new_afeat+=afeat
                new_vfeat += vfeat[:remainder]
                new_afeat += afeat[:remainder]
                vfeat = new_vfeat
                afeat = new_afeat
            else:
                vfeat = random.sample(vfeat,args.bagsize)
                afeat = random.sample(afeat,args.bagsize)
            neg_vfeat = torch.tensor(vfeat)
            neg_afeat = torch.tensor(afeat)
            pos_label = torch.ones(1,)
            neg_label = torch.zeros(1,)

            pos_vfeats.append(pos_vfeat)
            pos_afeats.append(pos_afeat)
            neg_vfeats.append(neg_vfeat)
            neg_afeats.append(neg_afeat)
            pos_labels.append(pos_label)
            neg_labels.append(neg_label)
        pos_vfeats = torch.stack(pos_vfeats)
        pos_afeats = torch.stack(pos_afeats)
        neg_vfeats = torch.stack(neg_vfeats)
        neg_afeats = torch.stack(neg_afeats)
        pos_labels = torch.stack(pos_labels)
        neg_labels = torch.stack(neg_labels)

        return pos_vfeats,pos_afeats,neg_vfeats,neg_afeats,pos_labels,neg_labels

#知视频的hihglight频率比长视频的高，那么为每个segment计算一个分值，则短视频的平均分值应该为较高
class MILDatasetv2(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        keys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 

        self.short_videos = []
        self.long_videos = []
        #2128327541805730762 11
        # pdb.set_trace()
        # print(keys)
        if args.paper_data_split:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper and self.duration[key]>args.short_lower: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower and self.duration[key]<args.long_upper:
                    self.long_videos.append(key)
        else:
            for key in tqdm(keys):
                if self.duration[key]<args.short_upper: 
                    self.short_videos.append(key)
                if self.duration[key]>args.long_lower:
                    self.long_videos.append(key)
        # pdb.set_trace()
        if len(self.short_videos) > len(self.long_videos):
            re = int(len(self.short_videos)/len(self.long_videos))
            remainder = len(self.short_videos)%len(self.long_videos)
            new_long_videos = []
            for i in range(re):
                new_long_videos+=self.long_videos
            new_long_videos += random.sample(self.long_videos, remainder)
            self.long_videos = new_long_videos
        else:
            re = int(len(self.long_videos)/len(self.short_videos))
            remainder = len(self.long_videos)%len(self.short_videos)
            new_short_videos = []
            for i in range(re):
                new_short_videos+=self.short_videos
            new_short_videos += random.sample(self.short_videos, remainder)
            self.short_videos = new_short_videos
        self.GeneratePairs()


    def GeneratePairs(self):
        
        random.shuffle(self.short_videos)
        random.shuffle(self.long_videos)
        self.pairs = list(zip(self.short_videos,self.long_videos))
        return

    def __len__(self):
        return len(self.short_videos) if len(self.short_videos) > len(self.long_videos) else len(self.long_videos)

    def __getitem__(self, idx):
        (posKey,negKey) = self.pairs[idx]
        video_seg = self.video_features[posKey]
        audio_seg = self.audio_features[posKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
       
        pos_vfeat = torch.tensor(vfeat)
        pos_afeat = torch.tensor(afeat)

        video_seg = self.video_features[negKey]
        audio_seg = self.audio_features[negKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
       
        neg_vfeat = torch.tensor(vfeat)
        neg_afeat = torch.tensor(afeat)

        return pos_vfeat,pos_afeat,neg_vfeat,neg_afeat

class MILDatasetv3(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

       
        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        commonKeys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 
        self.posKeys = []
        self.negKeys = []
        for key in tqdm(commonKeys):
            if self.duration[key]<=args.short_upper and self.duration[key]>=args.short_lower:
                self.posKeys.append(key)
            if self.duration[key]<=args.long_upper:
                self.negKeys.append(key)
        
        if len(self.negKeys) > len(self.posKeys):
            re = int(len(self.negKeys)/len(self.posKeys))
            remainder = len(self.negKeys)%len(self.posKeys)
            new_posKeys = []
            for i in range(re):
                new_posKeys+=self.posKeys
            new_posKeys += random.sample(self.posKeys, remainder)
            self.posKeys = new_posKeys
        else:
            re = int(len(self.posKeys)/len(self.negKeys))
            remainder = len(self.posKeys)%len(self.negKeys)
            new_negKeys = []
            for i in range(re):
                new_negKeys+=self.negKeys
            new_negKeys += random.sample(self.negKeys, remainder)
            self.negKeys = new_negKeys
        self.GeneratePairs()
    def GeneratePairs(self):
        #shuffle
        random.shuffle(self.posKeys)
        random.shuffle(self.negKeys)
        self.pairs = list(zip(self.posKeys,self.negKeys))
        return

    def __len__(self):
        return len(self.negKeys) if len(self.negKeys) > len(self.posKeys) else len(self.posKeys)

    def __getitem__(self, idx):
        (posKey,negKey) = self.pairs[idx]
        video_seg = self.video_features[posKey]
        audio_seg = self.audio_features[posKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        pos_vfeat = torch.tensor(vfeat)
        pos_afeat = torch.tensor(afeat)

        video_seg = self.video_features[negKey]
        audio_seg = self.audio_features[negKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat = random.sample(vfeat,args.bagsize)
            afeat = random.sample(afeat,args.bagsize)
        neg_vfeat = torch.tensor(vfeat)
        neg_afeat = torch.tensor(afeat)
        pos_label = torch.ones(1,)
        neg_label = torch.zeros(1,)


        return pos_vfeat,pos_afeat,neg_vfeat,neg_afeat,pos_label,neg_label

class MILDatasetv4(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        if args.dataset == 'youtube':
            self.domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
        if args.dataset == 'tvsum':
            self.domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        # self.domains.remove(domain)
        self.negBags_video = defaultdict(list)
        self.negBags_audio = defaultdict(list)
        self.negBags_duration = defaultdict(float)
        for bag in self.domains:
            duration = np.load(train_path+'/'+bag+'_duration.npy',allow_pickle=True).tolist()
            video_features = np.load(train_path+'/'+bag+'_1s.npy',allow_pickle=True).tolist()
            audio_features = np.load(train_path+'/'+bag+'_audio_edited_nopost.npy',allow_pickle=True).tolist()
            self.negBags_video = {**self.negBags_video,**video_features}
            self.negBags_audio = {**self.negBags_audio,**audio_features}
            self.negBags_duration = {**self.negBags_duration,**duration}
        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        commonKeys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 
        self.posKeys = []
        for key in tqdm(commonKeys):
            if self.duration[key]<=args.short_upper and self.duration[key]>=args.short_lower:
                self.posKeys.append(key)
        duration_keys = []
        for key in list(self.negBags_duration.keys()):
            if self.negBags_duration[key]>=args.long_lower:
                duration_keys.append(key)
        video_keys = list(self.negBags_video.keys())
        audio_keys = list(self.negBags_audio.keys())
        self.negKeys = list(set(duration_keys).intersection(set(video_keys).intersection(set(audio_keys))))
        self.availableKeys = copy.deepcopy(self.posKeys)
        
        if len(self.negKeys) > len(self.posKeys):
            re = int(len(self.negKeys)/len(self.posKeys))
            remainder = len(self.negKeys)%len(self.posKeys)
            new_posKeys = []
            for i in range(re):
                new_posKeys+=self.posKeys
            new_posKeys += random.sample(self.posKeys, remainder)
            self.posKeys = new_posKeys
        else:
            re = int(len(self.posKeys)/len(self.negKeys))
            remainder = len(self.posKeys)%len(self.negKeys)
            new_negKeys = []
            for i in range(re):
                new_negKeys+=self.negKeys
            new_negKeys += random.sample(self.negKeys, remainder)
            self.negKeys = new_negKeys
        self.GeneratePairs()
    def GeneratePairs(self):
        #shuffle
        random.shuffle(self.posKeys)
        random.shuffle(self.negKeys)
        self.pairs = list(zip(self.posKeys,self.negKeys))
        return

    def __len__(self):
        return len(self.negKeys) if len(self.negKeys) > len(self.availableKeys) else len(self.availableKeys)

    def __getitem__(self, idx):
        (posKey,negKey) = self.pairs[idx]
        video_seg = self.video_features[posKey]
        audio_seg = self.audio_features[posKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        pos_vfeat = torch.tensor(vfeat)
        pos_afeat = torch.tensor(afeat)

        video_seg = self.negBags_video[negKey]
        audio_seg = self.negBags_audio[negKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat = vfeat[:args.bagsize]
            afeat = afeat[:args.bagsize]
        neg_vfeat = torch.tensor(vfeat)
        neg_afeat = torch.tensor(afeat)
        pos_label = torch.ones(1,)
        neg_label = torch.zeros(1,)


        return pos_vfeat,pos_afeat,neg_vfeat,neg_afeat,pos_label,neg_label

class ADDataset(Dataset):
    def __init__(self, train_path,domain,num_per_group):
        super().__init__()
        if args.dataset == 'youtube':
            self.domains = ['dog','parkour','gymnastics','surfing','skating','skiing']
        if args.dataset == 'tvsum':
            self.domains = ['BK','BT','DS','FM','GA','MS','PK','PR','VT','VU']
        if args.dataset == 'cosum':
            self.domains = ['base_jump', 'bike_polo', 'eiffel_tower', 'excavators_river_cross', 'kids_playing_in_leaves', 
                            'MLB', 'NFL', 'notre_dame_cathedral', 'statue_of_liberty',  'surf']
        # self.train_path = train_path+'/'+domain
        self.num_per_group = num_per_group
        self.duration = np.load(train_path+'/'+domain+'_duration.npy',allow_pickle=True).tolist()
        self.video_features = np.load(train_path+'/'+domain+'_1s.npy',allow_pickle=True).tolist()
        self.audio_features = np.load(train_path+'/'+domain+'_audio_edited_nopost.npy',allow_pickle=True).tolist()

        self.domains.remove(domain)
        self.negBags_video = defaultdict(list)
        self.negBags_audio = defaultdict(list)
        self.negBags_duration = defaultdict(float)
        for bag in self.domains:
            duration = np.load(train_path+'/'+bag+'_duration.npy',allow_pickle=True).tolist()
            video_features = np.load(train_path+'/'+bag+'_1s.npy',allow_pickle=True).tolist()
            # pdb.set_trace()
            audio_features = np.load(train_path+'/'+bag+'_audio_edited_nopost.npy',allow_pickle=True).tolist()
            self.negBags_video = {**self.negBags_video,**video_features}
            self.negBags_audio = {**self.negBags_audio,**audio_features}
            self.negBags_duration = {**self.negBags_duration,**duration}
        keys = list(self.duration.keys())
        video_keys = list(self.video_features.keys())
        audio_keys = list(self.audio_features.keys())
        commonKeys = list(set(audio_keys).intersection(set(keys).intersection(set(video_keys)))) 
        self.posKeys = []
        for key in tqdm(commonKeys):
            if self.duration[key]<=args.short_upper and self.duration[key]>=args.short_lower:
                self.posKeys.append(key)
        duration_keys = []
        for key in list(self.negBags_duration.keys()):
            if self.negBags_duration[key]>=args.long_lower:
                duration_keys.append(key)
        video_keys = list(self.negBags_video.keys())
        audio_keys = list(self.negBags_audio.keys())
        self.negKeys = list(set(duration_keys).intersection(set(video_keys).intersection(set(audio_keys))))
        self.availableKeys = copy.deepcopy(self.posKeys)
        
        if len(self.negKeys) > len(self.posKeys):
            re = int(len(self.negKeys)/len(self.posKeys))
            remainder = len(self.negKeys)%len(self.posKeys)
            new_posKeys = []
            for i in range(re):
                new_posKeys+=self.posKeys
            new_posKeys += random.sample(self.posKeys, remainder)
            self.posKeys = new_posKeys
        else:
            re = int(len(self.posKeys)/len(self.negKeys))
            remainder = len(self.posKeys)%len(self.negKeys)
            new_negKeys = []
            for i in range(re):
                new_negKeys+=self.negKeys
            new_negKeys += random.sample(self.negKeys, remainder)
            self.negKeys = new_negKeys
        self.GeneratePairs()
    def GeneratePairs(self):
        #shuffle
        random.shuffle(self.posKeys)
        random.shuffle(self.negKeys)
        self.pairs = list(zip(self.posKeys,self.negKeys))
        return

    def __len__(self):
        return len(self.negKeys) if len(self.negKeys) > len(self.availableKeys) else len(self.availableKeys)

    def __getitem__(self, idx):
        (posKey,negKey) = self.pairs[idx]
        video_seg = self.video_features[posKey]
        audio_seg = self.audio_features[posKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat= vfeat[:args.bagsize]
            afeat= afeat[:args.bagsize]

            # vfeat = random.sample(vfeat,args.bagsize)
            # afeat = random.sample(afeat,args.bagsize)
        pos_vfeat = torch.tensor(vfeat)
        pos_afeat = torch.tensor(afeat)

        video_seg = self.negBags_video[negKey]
        audio_seg = self.negBags_audio[negKey]
        vfeat = []
        afeat = []
        for vseg,aseg in zip(video_seg,audio_seg):
            vfeat.append(vseg['features'])
            afeat.append(aseg['features'])
        if len(vfeat) <= args.bagsize:
            re = int(args.bagsize/len(vfeat))
            remainder = args.bagsize%len(vfeat)
            new_vfeat = []
            new_afeat = []
            for i in range(re):
                new_vfeat+=vfeat
                new_afeat+=afeat
            new_vfeat += vfeat[:remainder]
            new_afeat += afeat[:remainder]
            vfeat = new_vfeat
            afeat = new_afeat
        else:
            vfeat= vfeat[:args.bagsize]
            afeat= afeat[:args.bagsize]
        neg_vfeat = torch.tensor(vfeat)
        neg_afeat = torch.tensor(afeat)
        pos_label = torch.ones(1,)
        neg_label = torch.zeros(1,)


        return pos_vfeat,pos_afeat,neg_vfeat,neg_afeat,pos_label,neg_label


if __name__ == "__main__":
    feature = np.load('/home/share/Highlight/proDataset/DomainSpecific/feature/dog/msjK8nHZHZ0.mp4.npy').tolist()
    test_set = Test('/home/share/Highlight/proDataset/DomainSpecific','dog')
    # train_loader = DataLoader(test_set, shuffle=True, batch_size=16, pin_memory=True, num_workers=8,collate_fn = test_set.collate_fn)
    train_loader = DataLoader(test_set, shuffle=True, batch_size=16, pin_memory=True, num_workers=8)

    for batch_idx, feature,labels,names in enumerate(train_loader):
        print(names)
    # pdb.set_trace()
    # len(t)
    # just 4 testing dataset
