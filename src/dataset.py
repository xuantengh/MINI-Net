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
            inds = list(range(len(vfeat)))

            idxs = np.random.choice(inds,args.bagsize,replace = False)
            vfeat = np.array(vfeat)
            afeat = np.array(afeat)
            idxs = np.array(idxs)
            vfeat = vfeat[idxs]
            afeat = afeat[idxs]
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
            inds = list(range(len(vfeat)))
            idxs = np.random.choice(inds,args.bagsize,replace = False)
            vfeat = np.array(vfeat)
            afeat = np.array(afeat)
            idxs = np.array(idxs)
            vfeat = vfeat[idxs]
            afeat = afeat[idxs]
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
