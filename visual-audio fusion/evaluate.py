import numpy as np
import os
from tqdm import tqdm
from opts import args
import pdb

domain = args.domain
# proportion = args.proportion

def evaluate(machine_summary, GT,topK):
    human_annotation = []
    predicted_annotation = []
    keys = list(machine_summary.keys())
    for video in GT.keys():
        if video not in keys:
            continue
        x = machine_summary[video]
        annot = GT[video]
        h = []
        for at in annot:
            h.append(at['scores'])
        human_annotation.append(h)
        predicted_annotation.append(x)
    return evaluate_all(predicted_annotation, human_annotation, topK)

def TVsumOrCoSumEvaluate(machine_summary, GT,topK):
    human_annotation = []
    predicted_annotation = []
    keys = list(machine_summary.keys())
    for video in GT.keys():
        if video not in keys:
            continue
        x = machine_summary[video]
        annot = GT[video]
        h = []
        for at in annot:
            h.append(at['scores'])
        human_annotation.append(h)
        predicted_annotation.append(x)
    return TVsumOrCoSumEvaluate_all(predicted_annotation, human_annotation, topK)

def evaluate_sample(machine_summary, user_summary, topk=5):
    """Compare machine summary with user summary.
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    machine_summary： [1，n]
    user_summary：    [n_user,n_frames]
    """
    # machine_summary = machine_summary.astype(np.int16)
    # user_summary = user_summary.astype(np.int16)
    # pdb.set_trace()
    # pdb.set_trace()
    user_summary = np.array(user_summary)
    n_users, n_frames = user_summary.shape
    machine_summary = np.array(machine_summary)
    # binarization
    # machine_summary[machine_summary > 0] = 1
    # user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        user_summary = user_summary[:,:len(machine_summary)]
    sortIdx = np.argsort(-machine_summary)
    machine_summary = machine_summary[sortIdx]
    # pdb.set_trace()
    n_user_summary = []
    for i in range(n_users):
        temp = user_summary[i][sortIdx]
        n_user_summary.append(temp)
    user_summary = np.array(n_user_summary)
    n_users, n_frames = user_summary.shape

    APs = []
    avg_precision = []
    avg_recall = []
    # pdb.set_trace()
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        n_good = gt_summary.sum()
        ap = 0.0
        intersect_size = 0.0
        old_recall = 0.0
        old_precision = 1.0
        for j in range(n_frames):
            if gt_summary[j] == 1:
                intersect_size+=1
            recall = intersect_size/n_good
            precision = intersect_size / (j+1)
            ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision
        APs.append(ap)
        avg_precision.append(old_precision)
        avg_recall.append(old_recall)
    aps = np.array(APs)
    avg_precision = np.array(avg_precision)
    avg_recall = np.array(avg_recall)
    aps.sort()
    avg_precision.sort()
    avg_recall.sort()
    topk_mAP = aps[-topk:].mean()
    topk_pre = avg_precision[-topk:].mean()
    topk_rec = avg_recall[-topk:].mean()
    return topk_mAP, topk_pre, topk_rec

def TVsumOrCoSumEvaluate_sample(machine_summary, user_summary, topk=5):
    """Compare machine summary with user summary.
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    machine_summary： [1，n]
    user_summary：    [n_user,n_frames]
    """
    # machine_summary = machine_summary.astype(np.int16)
    # user_summary = user_summary.astype(np.int16)
    # pdb.set_trace()
    machine_summary = np.array(machine_summary)
    user_summary = np.array(user_summary)
    # print(user_summary.shape)
    n_users, n_frames = user_summary.shape
    # binarization
    # machine_summary[machine_summary > 0] = 1
    # user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])
    sortIdx = np.argsort(-machine_summary)
    machine_summary = machine_summary[sortIdx]
    n_user_summary = []
    for i in range(n_users):
        temp = user_summary[i][sortIdx]
        n_user_summary.append(temp)
    user_summary = np.array(n_user_summary)
    # pdb.set_trace()

    #选择topk

    user_summary = user_summary[:,:topk]

    APs = []
    avg_precision = []
    avg_recall = []
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx]
        n_good = gt_summary.sum()
        if n_good==0:
            APs.append(0)
            avg_precision.append(0)
            avg_recall.append(0)
            continue
        ap = 0.0
        intersect_size = 0.0
        old_recall = 0.0
        old_precision = 1.0
        for j in range(topk):
            if gt_summary[j] == 1:
                intersect_size+=1
            recall = intersect_size/n_good
            precision = intersect_size / (j+1)
            ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
            old_recall = recall
            old_precision = precision
        APs.append(ap)
        avg_precision.append(old_precision)
        avg_recall.append(old_recall)
    aps = np.array(APs)
    avg_precision = np.array(avg_precision)
    avg_recall = np.array(avg_recall)
    aps.sort()
    avg_precision.sort()
    avg_recall.sort()
    topk_mAP = aps.mean()
    topk_pre = avg_precision.mean()
    topk_rec = avg_recall.mean()
    return topk_mAP, topk_pre, topk_rec


def TVsumOrCoSumEvaluate_all(machine_summary, user_summary, topk=5):
    '''
    machine_summary [n_sample,n]
    user_summary [n_sample,n_user,n_frames]
    '''
    n_sample = len(user_summary)
    mAP = 0.0
    precision = 0.0
    recall = 0.0
    for i in tqdm(range(n_sample)):
        (tmp1, tmp2, tmp3) = TVsumOrCoSumEvaluate_sample(machine_summary[i], user_summary[i], topk)
        mAP += tmp1
        precision += tmp2
        recall += tmp3
    n_sample=n_sample+1e-9
    return mAP/n_sample, precision/n_sample, recall/n_sample


def evaluate_all(machine_summary, user_summary, topk=5):
    '''
    machine_summary [n_sample,n]
    user_summary [n_sample,n_user,n_frames]
    '''
    n_sample = len(user_summary)
    mAP = 0.0
    precision = 0.0
    recall = 0.0
    for i in tqdm(range(n_sample)):
        (tmp1, tmp2, tmp3) = evaluate_sample(machine_summary[i], user_summary[i], topk)
        mAP += tmp1
        precision += tmp2
        recall += tmp3
    n_sample=n_sample+1e-9
    return mAP/n_sample, precision/n_sample, recall/n_sample


if __name__ == '__main__':
    print('enter dataset: youtube | tvsum | summe?')
    dataset = input()
    summary(domain, dataset)