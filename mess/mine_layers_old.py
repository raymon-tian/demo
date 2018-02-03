#coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from mine_utils import cal_mask_4D,cal_mask_1D

class AttentionMap(nn.Module):

    def __init__(self,exp=1,method="max"):
        super(AttentionMap, self).__init__()
        self.exp = exp
        self.method = method

    def forward(self, fea_maps):
        """
        给定特征图，计算其对应的attention map
        :param fea_maps: Variable.cuda (N,C,H,W)
        :return: Variable.cuda (N,H,W)
        """
        fea_maps = torch.abs(fea_maps)
        fea_maps = torch.pow(fea_maps, self.exp)
        if self.method == "max":
            att_maps = torch.max(fea_maps, 1, False)[0]
        elif self.method == "sum":
            att_maps = torch.sum(fea_maps, 1, False)
        else:
            assert "method is illegal\n"
        N, H, W = att_maps.size()
        temp = att_maps.view((N, -1)) # (N,H*W)
        norm = torch.norm(temp, 2, 1, False).view((-1, 1))  # (N,1)
        norm = norm.expand_as(temp)  # (N,H*W) 不知道通过这种方式会不会破坏计算图
        temp = temp / (norm + 1e-16)  # 防止除0
        temp = temp.view((N, H, W))

        return temp

class ChannelMultiplier(nn.Module):
    """
    对于一个输入的特征图 X (N,C,H,W),在其每一个通道处学习一个scalar，一共学习得到C个scalar
    """
    def __init__(self,num_c):
        super(ChannelMultiplier,self).__init__()
        self.num_c = num_c
        self.multiplier = nn.Parameter(torch.randn(num_c))
    def forward(self, x):
        """
        计算weighted的特征图,在每一个通道处乘以一个scalar
        :param x: Variable cuda (N,C,H,W)
        :return: Variable cuda (N,C,H,W)
        """
        weights_x_list = []
        for i in range(self.num_c):
            mask = cal_mask_4D(x, 1, i)
            fea_map = x * mask
            fea_map = fea_map * self.multiplier.view((1,self.num_c,1,1)).expand_as(fea_map)
            weights_x_list.append(fea_map)
        weights_x = sum(weights_x_list)

        return weights_x


class SubspaceCluster(nn.Module):
    # """
    # 基于subspace聚类的思想，对特征图进行聚类，从而达到减少特征图数量的目的，进而达到剪枝特征图、卷积核的目的
    # 1. SubspaceCluster是作为卷基层的一种插件机制存在；当有特征图存在的时候，该种层才有存在的意义
    # 2. 使用自编码器的方式来进行Subspace中自表示矩阵的学习
    # 3. 不清楚其约束
    # 4. 不要在 子图内 求 loss
    # """
    def __init__(self,H,W,K):
        super(SubspaceCluster,self).__init__()
        self.H = H
        self.W = W
        self.K = K
        self.fc1 = nn.Linear(in_features=self.K,out_features=self.K,bias=False)

    def forward(self, X):
        """
        对输入特征图进行自表示学习
        :param x: (N,C,H,W)
        :return:
        """
        N,C,H,W = X.size()

        self.X = X.view((N, C, -1)).permute(0,2,1) #(N,H*W,C)
        self.X1 = self.fc1(self.X)  # (N,H*W,C)

        self.diff_recons = self.X - self.X1
        self.zeros_target_recons = Variable(self.diff_recons.data.new(self.diff_recons.data.size()).fill_(0.))

        self.zeros_target_norm = Variable(self.fc1.weight.data.new(self.fc1.weight.data.size()).fill_(0.))

        return X
    """
    def __init__(self,H,W,K):
        # 
        # :param H: 2D卷积层输出——特征图 (N,C,H,W)
        # :param W: 2D卷积层输出——特征图 (N,C,H,W)
        # :param K: 自表示方阵的行数，其实就是C
        # 
        super(SubspaceCluster,self).__init__()
        self.H = H
        self.W = W
        self.K = K
        self.fc1 = nn.Linear(in_features=self.H*self.W,out_features=self.K)
        self.fc2 = nn.Linear(in_features=self.K,out_features=self.K)
        self.fc3 = nn.Linear(in_features=self.K,out_features=self.H*self.W)
    """

    """
    def forward(self,X):

        N,C,H,W = X.size()
        ''' 去除掉spatial维度'''
        self.X = X.view((N,C,-1))
        self.X1 = self.fc1(self.X) # (N,C,K)
        self.X2 = self.fc2(self.X1) # (N,C,K)
        self.X3 = self.fc3(self.X2) # (N,C,H*W)

        self.diff_sc = self.X2 - self.X1
        self.zeros_target_sc = Variable(self.diff_sc.data.new(self.diff_sc.data.size()).fill_(0.))
        # loss_sc = F.mse_loss(self.diff_sc,self.zeros_target_sc)
        loss_sc = nn.MSELoss()(self.diff_sc,self.zeros_target_sc)

        self.diff_recons = self.X - self.X3
        self.zeros_target_recons = Variable(self.diff_recons.data.new(self.diff_recons.data.size()).fill_(0.))
        loss_recons = nn.MSELoss()(self.diff_recons,self.zeros_target_recons)

        fc2_params = list(self.fc2.parameters())[0]
        self.zeros_target_norm = Variable(fc2_params.data.new(fc2_params.data.size()).fill_(0.))
        loss_norm = nn.L1Loss()(fc2_params,self.zeros_target_norm)

        self.X3 = self.X3.view((N,C,H,W)) # (N,C,H,W)

        return self.X3,(loss_recons,loss_sc,loss_norm)
    """


