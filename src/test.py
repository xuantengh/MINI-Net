
import torch
import logging
import torch.optim as optim
# import opts
import os
import tools
import pdb
from torch.utils.data import DataLoader
from loss import *
import dataset 
from Recorder import Recorder, Drawer
from tools import *
import numpy as np
from opts import args
from tqdm import tqdm
from collections import defaultdict
from evaluate import *
import model
import loss
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def test():

    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path+'/gt_'+args.dataset+'.npy',allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())

    video_features = np.load(args.test_path+'/feature/'+args.domain+'_1s.npy',allow_pickle=True ).tolist()
    audio_features = np.load(args.test_path+'/feature/'+args.domain+'_audio_edited_nopost.npy',allow_pickle=True ).tolist()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            prefix = ve.split('.')[0]
            vfs = video_features[prefix]
            afs = audio_features[prefix]
            scores = []
            vfeat = []
            afeat = []

            for vf,af in zip(vfs,afs):
                vfeat.append(vf['features'])
                afeat.append(af['features'])
            if(len(afeat)==0):
                continue
            
            vfeat = torch.Tensor(vfeat).to(device).unsqueeze(0)
            afeat = torch.Tensor(afeat).to(device).unsqueeze(0)
            scores,logits = f(vfeat,afeat)
            # pdb.set_trace()
            summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
    mechine_summary = clip2frame(summary)
    mAP_1,pre,recall = evaluate(mechine_summary,category_dict,5)
    # mAP_2,pre,recall = evaluate(mechine_summary,category_dict,15)
    return mAP_1, pre, recall

def TVsumOrCoSumtest():

    summary = defaultdict(float)
    f.eval()
    gt_dict = np.load(args.test_path+'/gt_'+args.dataset+'.npy',allow_pickle=True).tolist()
    category_dict = gt_dict[args.domain]
    videos = list(category_dict.keys())

    video_features = np.load(args.test_path+'/feature/'+args.domain+'_1s.npy',allow_pickle=True ).tolist()
    audio_features = np.load(args.test_path+'/feature/'+args.domain+'_audio_edited_nopost.npy',allow_pickle=True ).tolist()
    # pdb.set_trace()
    with torch.no_grad():
        for ve in tqdm(videos):
            label = category_dict[ve]
            prefix = ve.split('.')[0]
            vfs = video_features[prefix]
            afs = audio_features[prefix]
            scores = []
            vfeat = []
            afeat = []
            for vf,af in zip(vfs,afs):
                vfeat.append(vf['features'])
                afeat.append(af['features'])
            if(len(afeat)==0):
                continue
            vfeat = torch.Tensor(vfeat).to(device).unsqueeze(0)
            afeat = torch.Tensor(afeat).to(device).unsqueeze(0)
            scores,logits = f(vfeat,afeat)
            # pdb.set_trace()
            summary[ve] = scores.cpu().numpy().reshape(-1).tolist()
    
    mechine_summary = clip2segment(summary,category_dict)
    mAP_5, pre, rec = TVsumOrCoSumEvaluate(mechine_summary,category_dict,5)
    # mAP_2, _, __ = TVsumOrCoSumEvaluate(mechine_summary,category_dict,15)
    return mAP_5, pre, rec


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # loader_dict = dict(shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    # dataset = getattr(dataset,args.DS)(args.train_path,args.domain,args.num_per_group)
    # train_loader = DataLoader(dataset, **loader_dict)
    model_path = "./model_param/{}_{}_max.pth".format(args.dataset, args.domain)
    checkpoint = torch.load(model_path)
    f = getattr(model,args.FNet)(AM=getattr(model,args.AM)).to(device)
    f.load_state_dict(checkpoint["fNet"])
    # CELoss = torch.nn.CrossEntropyLoss()
    # AHLoss = getattr(loss,args.AHLoss)().to(device)

    # rankingloss = Rankingloss().to(device)
    # recoder = Recorder('{}_{}'.format(args.dataset, args.domain))
    # tools.mkdir_if_missed('./model_param')
    # mAP = 0
    # maxMap = -1
    # opt = optim.SGD(f.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # for epoch_idx in range(args.epoch):
    #     train(epoch_idx,mAP)
    #     if epoch_idx%args.interval == 0:
    #         print('========================testing=============================')
    #         if args.dataset!='youtube':
    #             mAP,pre,recall = TVsumOrCoSumtest()
    #         else:
    #             mAP,pre,recall = test()
    #         if mAP>maxMap:
    #             torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_max.pth'.format(args.dataset, args.domain)))
    #             maxMap = mAP
    #         torch.save({'fNet':f.state_dict(),'args':args}, os.path.join('./model_param', '{}_{}_final.pth'.format(args.dataset, args.domain)))
            
    #         # print(mAP)
    #     dataset.GeneratePairs()
    #     train_loader = DataLoader(dataset, **loader_dict)
    mAP_5 = 0.0
    # mAP_15 = 0.0
    if args.dataset!='youtube':
        mAP_5, pre, recall = TVsumOrCoSumtest()
    else:
        mAP_5, pre, recall = test()
    print("{}\t{}, \t mAP_5: {:.4f}".format(args.dataset, args.domain, mAP_5))

    # Drawer('../fig/{}_{}.pkl'.format(dataset, domain),
        #    '../fig/{}_{}_train_loss.png'.format(dataset, domain))





