# implement two networks:
# 1. f(x): ranking function
# 2. h(xi, xj): checking function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from opts import args
from torch.nn import init

#multi head,dropout
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
        output =self.dropout(self.relu(self.fc2(output)))
        return output
class AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
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

#multi head,dropout, 最后没有dropout
class AttentionUnit_1(nn.Module):
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
class AttentionModule_1(nn.Module):
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

#multi head
class Fusion_AttentionUnit(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.afc = nn.Linear(in_feature_dim,64)
        self.vfc = nn.Linear(in_feature_dim,64)
        self.fusion = nn.Linear(64,out_feature_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.dropout3 = nn.Dropout(0.5)

 

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        # x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        vx = self.dropout1(self.relu(self.vfc(vx)))
        ax = self.dropout2(self.relu(self.afc(ax)))
        fx = self.relu(self.fusion(vx+ax))
        return fx
class Fusion_AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(Fusion_AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat

#multi head
class Point_AttentionUnit(nn.Module):
    def __init__(self,in_feature_dim = 128, out_feature_dim = 32):
        super().__init__()
        self.subspace = 64
        self.Q = nn.Linear(in_feature_dim,self.subspace)
        self.W = nn.Linear(in_feature_dim,self.subspace)
        self.P = nn.Linear(self.subspace,1)

        self.V = nn.Linear(in_feature_dim,out_feature_dim)
        # self.dropout3 = nn.Dropout(0.5)

        self.relu = nn.ReLU()
    def forward(self,vx,ax):
        # x = torch.cat((vx,ax),-1)  #concat both visual and audio features
        sigma = self.relu(self.P(self.Q(vx)+self.W(ax)))
        fx = sigma*self.V(ax)
        return fx
class Point_AttentionModule(nn.Module):
    def __init__(self,feature_dim=128,num_of_subAttention=4):
        super().__init__()
        self.attentionUnits = nn.ModuleList()
        self.num_of_subAttention = num_of_subAttention
        # self.Conv = torch.nn.Sequential()
        # self.Conv.add_module('conv_1',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) # [?,224,224,10]
        # self.Conv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=int(num_of_subAttention/2),stride=int(num_of_subAttention/2))) #[?,112,112,10]
        # self.Conv.add_module('conv_2',torch.nn.Conv2d(feature_dim,feature_dim,kernel_size=3,stride=1,padding=1)) #[?,112,112,32]
        # self.Conv.add_module('pool',nn.AdaptiveAvgPool2d((1,1)))
        for N in range(self.num_of_subAttention):
            self.attentionUnits.append(Point_AttentionUnit(feature_dim,int(feature_dim/num_of_subAttention)))
        self.relu = nn.ReLU()

    def forward(self,vx,ax):
        isFirst=True
        for N in range(self.num_of_subAttention):
            if(isFirst):
                concat = self.attentionUnits[N](vx,ax)
                isFirst=False
            else:
                concat = torch.cat((concat,self.attentionUnits[N](vx,ax)),1)
        #attention fusion
        # attention = F.softmax(torch.log(concat+1e-9),1)
        # attention = conca
        # vx_att = vx+concat
        return concat

#一个AM，audio-guide attention
class FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        
        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  



#一个AM，audio-guide attention
class softmax_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 2)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 2)
        self.afc3 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        
        att = self.att_net(vo1,ax)
        att_x = att+vo1
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.softmax(self.attfc3(att_out))[:,1].view(-1,1).contiguous()

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.softmax(self.vfc3(vout))[:,1].view(-1,1).contiguous()

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.softmax(self.afc3(aout))[:,1].view(-1,1).contiguous()
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class attention_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.WV = nn.Linear(128, 128)
        self.WA = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        
        att = self.att_net(vo1,ax)
        # att = self.sigmoid(att)
        visual_att_x = att+vo1
        audio_att_x = att+ax
        att_x = self.relu(self.WV(visual_att_x)+self.WA(audio_att_x))
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#一个AM，audio-guide attention
class AFNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.att_net = AM(128)
        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        
        att = self.att_net(vo1,ax)
        att_x = att+ax
        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout2(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)

        aout = self.dropout2(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  


#两个AM, dual-guide attention
class Dual_FNet(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.fusion = nn.Linear(128*2, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.visual_attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.visual_attfc3 = nn.Linear(64, 1)

        self.audio_attfc2 = nn.Linear(128, 64)
        self.audio_attfc3 = nn.Linear(64, 1)

        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax =  self.relu(self.afc(ax))

        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        # att_x = self.dropout5(self.relu(self.fusion(torch.cat((visual_att_x,audio_att_x),1))))

        visual_att_out = self.dropout2(self.relu(self.visual_attfc2(visual_att_x)))
        visual_att_out = self.visual_attfc3(visual_att_out)

        audio_att_out = self.dropout2(self.relu(self.audio_attfc2(audio_att_x)))
        audio_att_out = self.audio_attfc3(audio_att_out)

        vout = self.dropout3(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)
        aout = self.dropout4(self.relu(self.afc2(ax)))
        aout = self.afc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return visual_att_out,audio_att_out,vout,aout,visual_att_x,audio_att_x,vo1,ax  # [n, 8,2]  

#两个AM, dual-guide attention,统一回归器
class Dual_FNet_cat(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.fusion = nn.Linear(128*2, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        att_x = self.dropout5(self.relu(self.fusion(torch.cat((visual_att_x,audio_att_x),1))))

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout3(self.relu(self.attfc2(vo1)))
        vout = self.attfc3(vout)
        aout = self.dropout4(self.relu(self.attfc2(ax)))
        aout = self.attfc3(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  

#两个AM, dual-guide attention,统一回归器
class Dual_FNet_add(nn.Module):
    def __init__(self, feature_dim = 512,AM=None):
        super().__init__()
        self.visual_att_net = AM(128)
        self.audio_att_net = AM(128)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.afc = nn.Linear(128, 128)
        self.WA = nn.Linear(128, 128)
        self.WV = nn.Linear(128, 128)

        self.fusion = nn.Linear(128, 128)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.attfc2 = nn.Linear(128, 64)
        
        self.dropout2 = torch.nn.Dropout(0.5)
        self.attfc3 = nn.Linear(64, 1)
        self.vfc2 = nn.Linear(128, 64)
        self.afc2 = nn.Linear(128, 64)
        self.vfc3 = nn.Linear(64, 1)
        self.afc3 = nn.Linear(64, 1)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.dropout6 = torch.nn.Dropout(0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, vx,ax):
        # x [n, 8, 512]
        # x = x.view(-1, 512)
        vo1 = self.dropout1(self.relu(self.fc1(vx)))
        ax = self.relu(self.afc(ax))
        visual_att = self.visual_att_net(vo1,ax)
        visual_att_x = visual_att+vo1
        audio_att = self.audio_att_net(ax,vo1)
        audio_att_x = audio_att+ax

        att_x = self.dropout5(self.relu(self.fusion(self.WV(visual_att_x)+self.WA(audio_att_x))))

        att_out = self.dropout2(self.relu(self.attfc2(att_x)))
        att_out = self.attfc3(att_out)

        vout = self.dropout3(self.relu(self.vfc2(vo1)))
        vout = self.vfc3(vout)
        aout = self.dropout4(self.relu(self.afc2(ax)))
        aout = self.afc2(aout)
        # out = self.softmax(out)[:,1].contiguous()
        return att_out,vout,aout,att_x,vo1,ax  # [n, 8,2]  


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
# score without softmax
class MILModel9(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(MILModel9, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

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
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.relu(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
        scores = torch.transpose(scores,2, 1)  # KxN
        scores_noso = scores
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
        scores_noso = scores_noso.squeeze(1)
        scores = scores.squeeze(1)
        return scores_noso,logits

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

class Gated_Attention(nn.Module):
    def __init__(self,visual_feat_dim=512,audio_feat_dim=128,AM=None,**args):
        super(Gated_Attention, self).__init__()
        self.att_net = AM(128)
        self.vfc = nn.Linear(visual_feat_dim, audio_feat_dim)
        self.afc = nn.Linear(audio_feat_dim, audio_feat_dim)

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
        vo1 = self.dropout(self.relu(self.vfc(vfeat))).view(-1,self.subspace_dim)
        ax = self.relu(self.afc(afeat)).view(-1,self.subspace_dim)
        
        attfeat = self.att_net(vo1,ax)
        ffeat = attfeat+vo1
        ffeat = ffeat.view(b,n,-1)
        # ffeat = self.relu(self.fusionFc(catfeat)) #4096*32*128
        scores = self.W(self.tanh(self.V(ffeat))*self.sigmoid(self.U(ffeat)))
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
