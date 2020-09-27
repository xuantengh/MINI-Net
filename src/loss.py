import torch
import torch.nn as nn
import torch.nn.functional as FF
import pdb
from opts import args
from torch.nn import init
class LIMloss(nn.Module):
    def __init__(self):
        super(LIMloss, self).__init__()
        # self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        # self.margin = torch.Tensor([1], self.device)
        # self.alpha = torch.Tensor([0], self.device)

    def forward(self, fxi, fxj, w,margin=1):
        fxi = fxi.view(-1,args.num_per_group)
        fxj = fxj.view(-1,args.num_per_group)
        self.margin = torch.tensor([margin], dtype=torch.float32,requires_grad=False).to(self.devices)
        loss = torch.mul(w, torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj))).sum(1).mean() #[n,8]

        return loss
class Rankingloss(nn.Module):
    def __init__(self,margin=1):
        super(Rankingloss,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,fxi,fxj,margin=1):  
        # pdb.set_trace()
        self.margin = torch.tensor([margin], dtype=torch.float32,requires_grad=False).to(self.devices)
        # pdb.set_trace()
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, fxi), fxj)).mean()
        return loss

class AdaptiveHuberLoss(nn.Module):
    def __init__(self,margin=1,delta = 0.5):
        super(AdaptiveHuberLoss,self).__init__()
        self.delta = delta
        self.margin = margin
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,si,sj):
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, si), sj))
        mu = self.margin-si+sj
        certier = mu<self.delta
        not_certier = mu>=self.delta
        l1 = self.delta*torch.abs(loss)-1/2*self.delta*self.delta
        l2 = 1/2*torch.pow(loss,2)
        l1 = l1*not_certier.float()
        l2 = l2*certier.float()
        loss = l1+l2
        return loss.mean()
class TanHHuberLoss(nn.Module):
    def __init__(self,margin=1,delta = 0.25,gamma=0.5):
        super(TanHHuberLoss,self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.margin = margin
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,si,sj):
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, si), sj))
        mu = self.margin-si+sj
        outlier = mu<=self.delta
        inlier = mu>=self.gamma
        # ambiguity = mu>self.delta and mu<self.gamma
        l1 = self.delta*loss-1/2*self.delta*self.delta
        l2 = 1/2*torch.pow(loss,2)
        l1 = l1*outlier.float()
        l2 = l2*inlier.float()
        # l3 = loss*ambiguity.float()
        loss = l1+l2
        return loss.mean()


class ExponentialLoss(nn.Module):
    def __init__(self,margin=1,delta = 1.5):
        super(ExponentialLoss,self).__init__()
        self.delta = delta
        self.margin = margin
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.e = 0.25
        self.s = 2
    
    def forward(self,si,sj,epoch):
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, si), sj))
        lambda_ = self.s - (self.s-self.e)*epoch/args.epoch
        loss = torch.pow(loss,lambda_)
        return loss.mean()

class FusionLoss(nn.Module):
    def __init__(self,margin=1,delta = 1.5):
        super(FusionLoss,self).__init__()
        self.delta = delta
        self.margin = margin
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.margin = torch.tensor([1], dtype=torch.float32,requires_grad=False).to(self.devices)
        self.e = 0.25
        self.s = 2
    
    def forward(self,si,sj,epoch):
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, si), sj))
        lambda_ = self.s - (self.s-self.e)*epoch/args.epoch
        loss = torch.max(self.alpha, torch.add(torch.sub(self.margin, si), sj))
        mu = self.margin-si+sj
        certier = mu<self.delta
        not_certier = mu>=self.delta
        l1 = self.delta*torch.abs(loss)-1/2*self.delta*self.delta
        l2 = 1/2*torch.pow(loss,2)
        l1 = l1*not_certier.float()
        l2 = l2*certier.float()
        loss = l1+l2
        loss = torch.pow(loss,lambda_)
        return loss.mean()
class AdaptiveHingerLoss(nn.Module):
    def __init__(self,margin=1):
        super(AdaptiveHingerLoss,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,poss,negs,margin=1):  
        poss = torch.max(poss,1)[0].view(-1,1)
        negs = torch.max(negs,1)[0].view(-1,1)
        self.margin = torch.tensor([margin], dtype=torch.float32,requires_grad=False).to(self.devices)
        loss = torch.max(self.alpha, 1-poss+negs).mean()
        return loss

class AdaptiveHingerLossv2(nn.Module):
    def __init__(self,margin=1):
        super(AdaptiveHingerLossv2,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,poss,negs,margin=1):  
        poss = torch.max(poss,1)[0].view(-1,1)
        negs = torch.min(negs,1)[0].view(-1,1)
        self.margin = torch.tensor([margin], dtype=torch.float32,requires_grad=False).to(self.devices)
        loss = torch.max(self.alpha, 1-poss+negs).mean()
        return loss


class AdaptiveHingerLossv3(nn.Module):
    def __init__(self,margin=1):
        super(AdaptiveHingerLossv3,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,poss,negs,margin=1):  
        b,n = poss.shape
        poss = torch.max(poss,1)[0].expand(b,n)
        self.margin = torch.tensor([margin], dtype=torch.float32,requires_grad=False).to(self.devices)
        loss = torch.max(self.alpha, 1-poss+negs).mean(1).mean()
        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self,margin=1):
        super(SmoothnessLoss,self).__init__()
        self.devices = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.alpha = torch.tensor([0], dtype=torch.float32,requires_grad=False).to(self.devices)

    def forward(self,scores):  
        loss = torch.pow(scores[:-1]-scores[1:],2).sum()
        return loss

class SparsityLoss(nn.Module):
    def __init__(self):
        super(SparsityLoss,self).__init__()
    def forward(self,scores):
        return scores.sum()