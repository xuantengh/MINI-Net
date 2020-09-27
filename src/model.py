# implement two networks:
# 1. f(x): ranking function
# 2. h(xi, xj): checking function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from opts import args
from torch.nn import init

#multi head,dropout, 最后没有dropout
class AttentionUnit(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.fc1 = nn.Linear(2*in_feature_dim,in_feature_dim)
        self.fc2 = nn.Linear(in_feature_dim,out_feature_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        output = self.dropout(self.relu(self.fc1(x)))
        output =self.relu(self.fc2(output))
        return output
class AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit_1(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        return concat

class MILModel_vision(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel_vision, self).__init__()
        # self.att_net = AM(128)
        self.vfc = nn.Sequential(nn.Linear(visual_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        )
        

        # self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        ffeat = self.dropout(self.relu(self.vfc(vfeat)))
        # ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        # attfeat = self.att_net(vo1,ax)
        # ffeat = attfeat+vo1
        # ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
     
        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel_audio(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel_audio, self).__init__()
        # self.att_net = AM(128)
        self.afc = nn.Sequential(nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        nn.ReLU(),
        nn.Linear(audio_feat_dim, audio_feat_dim),
        )
        

        # self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        # ffeat = self.dropout(self.relu(self.vfc(vfeat)))
        ffeat = self.relu(self.afc(afeat))
        
        # attfeat = self.att_net(vo1,ax)
        # ffeat = attfeat+vo1
        # ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
     
        if self.training is True:
            scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits

class MILModel10(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel10, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.subspace_dim = 128
        self.fusionFc = nn.Linear(self.visual_feat_dim+self.audio_feat_dim, self.subspace_dim)
        # self.U = nn.Linear(self.subspace_dim, 64)
        self.V = nn.Linear(self.subspace_dim, 64)
        self.W = nn.Linear(64, 1)
        self.classifier = nn.Linear(self.subspace_dim,2)
        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,vfeat,afeat):
        # pdb.set_trace()
        b,n,d = vfeat.shape
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        
        # if self.training is True:
        scores = F.softmax(scores,2)
        # else:
        #     pdb.set_trace()
        zfeat = [torch.mm(s,f) for (s,f) in zip(scores,ffeat)]
        # pdb.set_trace()
        zfeat = torch.stack(zfeat)
        zfeat = zfeat.squeeze(1)
        # zfeat = torch.mm(scores,ffeat)
        logits = self.classifier(zfeat)
        scores = scores.squeeze(1)
        return scores,logits
