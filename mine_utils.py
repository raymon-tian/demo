#coding=utf-8
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans

def cal_attention_map(fea_maps, exp=1,method="max"):
    """
    给定特征图，计算其对应的attention map
    :param fea_maps: Variable.cuda (N,C,H,W)
    :return: Variable.cuda (N,H,W)
    """
    fea_maps = torch.abs(fea_maps)
    fea_maps = torch.pow(fea_maps, exp)
    if method == "max":
        att_maps = torch.max(fea_maps,1,False)[0]
    elif method == "sum":
        att_maps = torch.sum(fea_maps,1,False)
    N,H,W = att_maps.size()
    temp = att_maps.view((N,-1)) # (N,H*W)
    norm = torch.norm(temp,2,1,False).view((-1,1)) # (N,1)
    norm = norm.expand_as(temp) # (N,H*W) 不知道通过这种方式会不会破坏计算图
    temp = temp / (norm+1e-8) # 防止除0
    temp = temp.view((N,H,W))

    return temp

def cal_mask_4D(fea_maps,dim,idx):
    """
    给定一个特征图，计算对应的mask,使得在dim维度，idx处的取值为1
    :param fea_maps: Variable.cuda (N,C,H,W)
    :param dim: int 0 1 2 3
    :param idx: int
    :return: mask (N,C,H,W)
    """
    mask = Variable(torch.zeros(fea_maps.size()),requires_grad = False).cuda()
    if dim == 0:
        mask[idx,:,:,:] = 1.
    elif dim == 1:
        mask[:,idx,:,:] = 1.
    elif dim == 2:
        mask[:,:,idx,:] = 1.
    elif dim == 3:
        mask[:,:,:,idx] = 1.
    else:
        raise ValueError

    return mask

def cal_mask_1D(vec,idx):
    mask = Variable(torch.zeros(vec.size())).cuda()
    mask[idx] = 1.
    return mask

def inject_params(target,refer):
    """
    将预训练好的模型参数注射到现有的模型
    :param target: OrderedDict
    :param refer: OrderedDict
    :return:OrderedDict
    """
    for k in refer.keys():
        if k in target.keys():
            assert target[k].size() == refer[k].size(),"数据尺度不匹配"
            target[k] = refer[k]
    return target

def get_top_k(a,k):
    """
    返回一个list中的top-k元素，以及其对应的索引
    :param elements:list
    :return:list,list
    """
    top_k_idx = sorted(range(len(a)), key=lambda i: a[i])[-k:]
    top_k = [a[i] for i in top_k_idx]
    return top_k,top_k_idx

def check_shape(s1,s2):
    """
    检查两个list是否完全一致
    :param s1:
    :param s2:
    :return:
    """
    pass

def spectrum_cluster(self_rep_matrix,k):
    """
    谱聚类算法：返回各个样本的聚类类标
    :param self_rep_matrix: 自表示系数矩阵 numpy array (N,N)
    :param k: 聚类数目
    :return:[(idx1,idx2,...),...]
    """
    assert self_rep_matrix.shape[0] >= k, "聚类簇数不合法"
    C = self_rep_matrix
    W = np.abs(C) + np.abs(C.T)
    diag_vs = np.sum(W,axis=1)
    D1 = np.diag(np.float_power(diag_vs,-0.5))
    D2 = np.diag(np.float_power(diag_vs, 0.5))
    L = D1.dot(W).dot(D2)
    ''' 求出L的特征值以及特征向量'''
    w, v = np.linalg.eig(L)
    indics = np.argsort(w)[:k]
    v = v[:,indics] # (N,K)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(v)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)

    result = []
    for l in unique_labels:
        temp = []
        for i in range(len(labels)):
            if labels[i] == l:
                temp.append(i)
        result.append(temp)
    return result
