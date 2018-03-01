#coding=utf-8
"""
可以一次性学习完成所有层的稀疏向量
可以一次性剪枝所有的层
当然也可以逐步地剪枝某一层
需要的先验信息
{ (name_layer_i,name_layer_i+1),...(name_layer_i,name_layer_i+1) }
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch import nn
import os

from config import config
from prune_models import ChannelPruneNet
from data_loader import get_data_loader

base_path = os.path.join('./weight',config['dataset_name'] + '-' + config['model_name'],config['channel_select_algo'])
save_path = os.path.join(base_path,config['exp_name'])
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(os.path.join(save_path,'log-{}.txt'.format(config['phase'])), 'a'))
print = logger.info
print(config)

assert config['phase'] == 3, "phase error"

def train(e):

    global stu_model
    global lr
    global optimizer
    global criterion_cls
    global train_loader
    global epoch_average_loss

    ''' 初始化工具层 '''
    stu_model.train()
    epoch_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):

        ''' 1. 包裹Tensor为Variable '''
        stu_data, target = Variable(data.cuda(),requires_grad=True),Variable(target.cuda())#本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        ''' 2. 优化器梯度清零 '''
        optimizer.zero_grad()
        ''' 3. 前向传播 '''
        stu_sample_scores, stu_fea_maps = stu_model.forward(stu_data)
        ''' 3.1 loss 计算'''
        loss = criterion_cls(stu_sample_scores,target)
        print(
            'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:15.6f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),loss.data[0]))

        ''' 4. 误差反向传播 '''
        loss.backward()
        ''' 5. 参数更新'''
        optimizer.step()

        epoch_loss.append(loss.data[0])

    the_epoch_av_loss = sum(epoch_loss)/len(epoch_loss)
    epoch_average_loss.append(the_epoch_av_loss)

    """ 调整 lr """
    old_lr = lr
    lr = cust_adjust_lr(epoch_average_loss, lr)
    if old_lr != lr:
        optimizer = optim.Adam(stu_model.parameters(), lr=lr)

    print ('Epoch: {:3d}\taverage_epoch_loss: {:.6f}'.format(e,the_epoch_av_loss))

def test():

    global stu_model
    global test_loader

    stu_model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    for batch_idx,(data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, conv_output = stu_model.forward(data,is_test=True)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        _, pred_top1 = torch.topk(source=output.data,k=1,dim=1)
        _, pred_top3 = torch.topk(source=output.data,k=3,dim=1)
        _, pred_top5 = torch.topk(source=output.data,k=5,dim=1)
        count_top1 = 0
        count_top3 = 0
        count_top5 = 0
        target = target.unsqueeze(1)
        for temp_idx in range(target.size()[0]):
            if (target.data[temp_idx,0] in pred_top1[temp_idx]) is True:
                count_top1 += 1
            if (target.data[temp_idx,0] in pred_top3[temp_idx]) is True:
                count_top3 += 1
            if (target.data[temp_idx,0] in pred_top5[temp_idx]) is True:
                count_top5 += 1
        correct_top1 += count_top1
        correct_top3 += count_top3
        correct_top5 += count_top5

        print('{:5d}/{:5d}'.format( batch_idx * len(data), len(test_loader.dataset)))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, top1: {}/{} ({:.5f}%)\ttop3: {}/{} ({:.5f}%)\ttop5: {}/{} ({:.5f}%)\n'.format(
        test_loss,
        correct_top1, len(test_loader.dataset),100. * correct_top1 / len(test_loader.dataset),
        correct_top3, len(test_loader.dataset),100. * correct_top3 / len(test_loader.dataset),
        correct_top5, len(test_loader.dataset),100. * correct_top5 / len(test_loader.dataset)
    ))

def cust_adjust_lr(loss_his,lr):
    if len(loss_his) < 3:
        return lr
    if (loss_his[-1] > loss_his[-2]) and (loss_his[-2] > loss_his[-3]) and (lr > 0.25e-5):
        lr *= 0.5
        print('lr : {}'.format(lr))
        return lr
    elif lr < 0.25e-5:
        exit()
    else:
        return lr

if __name__ == '__main__':

    lr = config['lr']
    ''' 加载数据集 '''
    train_loader, test_loader = get_data_loader()
    ''' 初始化 student model '''
    stu_model = ChannelPruneNet(
        model_name=config['model_name'],
        channel_select_algo=config['channel_select_algo'],
        is_teacher=False,
        phase=config['phase'],
        weight_path=config['stu_weight_path'],
        conv_names=config['conv_pruned_names'],
    )
    stu_model.cuda()

    ''' 优化参数设置 ： 仅仅优化student model需要优化的参数 '''
    optimizer = optim.Adam(stu_model.parameters(), lr=lr)

    criterion_cls = nn.CrossEntropyLoss()

    epoch_average_loss = []

    for e in range(config['start_epoch'], config['epoch'] + 1):
        if e % config['test_freq'] == 0:
            test()
        train(e)
        if e % config['save_freq'] == 0 or e == config['epoch']:
            epoch_save_path = os.path.join(save_path, 'stage{}_epoch{}.pth'.format(config['phase'], e))
            torch.save(stu_model.state_dict(), epoch_save_path)
            pass
        if e % config['resample_data_freq'] == 0:
            train_loader = get_data_loader(only_train=True)