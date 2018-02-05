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
from mine_layers import AttentionMap

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

''' 加载数据集 '''
train_loader, test_loader = get_data_loader()
''' 初始化 teacher model '''
tea_model = ChannelPruneNet(
    model_name=config['model_name'],
    weight_path=config['weight_path'])
tea_model.cuda()
tea_model.train()
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
stu_model.train()
''' 优化参数设置 ： 仅仅优化student model需要优化的参数 '''
# optimizer = optim.Adam(list(stu_model.conv1.parameters())+list(stu_model.cm1.parameters()),lr=config['lr'])
params_opt = []
if config['phase'] == 1:
    # edit 这里可以设置成为自定义的层
    for conv_name in stu_model.all_conv_names:
        if config['channel_select_algo'] == 'sparse_vec':
            conv_layer_cm = eval('stu_model.'+conv_name+'_cm')
            params_opt.append({'params':conv_layer_cm.parameters()})
        elif config['channel_select_algo'] == 'subspace_cluster':
            conv_layer_sc = eval('stu_model.'+conv_name+'_sc')
            params_opt.append({'params':conv_layer_sc.parameters()})
        else:
            raise ValueError
elif config['phase'] == 2:
    for item in config['conv_pruned_names']:
        conv_name1, conv_name2, ratio = item
        conv_layer1 = eval('stu_model.' + conv_name1)
        conv_layer2 = eval('stu_model.' + conv_name2)
        if config['channel_select_algo'] == 'sparse_vec':
            params_opt.append({'params': conv_layer2.parameters()})
        elif config['channel_select_algo'] == 'subspace_cluster':
            ''' edit 这个地方需要考虑 '''
            params_opt.append({'params': conv_layer1.parameters()})
            params_opt.append({'params': conv_layer2.parameters()})
        else:
            raise ValueError
elif config['phase'] == 3:
    params_opt.append({'params': stu_model.parameters()})
else:
    raise ValueError

"""
for item in config['conv_pruned_names']:
    conv_name1,conv_name2,ratio = item
    conv_layer1 = eval('stu_model.'+conv_name1)
    conv_layer2 = eval('stu_model.'+conv_name2)
    # params_opt.append({'params': conv_layer1.parameters()})
    # params_opt.append({'params': conv_layer2.parameters()})
    if config['phase'] == 1:
        if config['channel_select_algo'] == 'sparse_vec':
            conv_layer1_cm = eval('stu_model.'+conv_name1+'_cm')
            params_opt.append({'params':conv_layer1_cm.parameters()})
        elif config['channel_select_algo'] == 'subspace_cluster':
            conv_layer1_sc = eval('stu_model.'+conv_name1+'_sc')
            params_opt.append({'params':conv_layer1_sc.parameters()})
    else:
        if config['channel_select_algo'] == 'sparse_vec':
            params_opt.append({'params':conv_layer2.parameters()})
        elif config['channel_select_algo'] == 'subspace_cluster':
            params_opt.append({'params':conv_layer2.parameters()})
"""

optimizer = optim.Adam(params_opt,lr=config['lr'])
''' 初始化工具层 '''
att_map_layer = AttentionMap()
criterion = nn.MSELoss()
criterion_l1 = nn.L1Loss(size_average=False) # 使用L1 Loss来计算l1 范数
criterion_cls = nn.CrossEntropyLoss().cuda()


# 打印cm层学到的参数
# for param in stu_model.cm1.parameters():
#     print param.data

def test():
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

# for e in range(1,config['epoch']+1):
def train(e):
    stu_model.train()
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        ''' 1. 包裹Tensor为Variable '''
        tea_data, _ = Variable(data.cuda(),volatile=True),Variable(target.cuda())#本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        stu_data, target = Variable(data.cuda(),requires_grad=True),Variable(target.cuda())#本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        ''' 2. 优化器梯度清零 '''
        optimizer.zero_grad()
        ''' 3. 前向传播 '''
        tea_sample_scores, tea_fea_maps = tea_model.forward(tea_data) # list ; teacher model 仅仅需要 inference，不需要求梯度
        stu_sample_scores, stu_fea_maps = stu_model.forward(stu_data)
        ''' 3.1 loss 计算'''
        if config['phase'] == 1:
            loss_reg_list = []
            loss_sc_recons_list = []
            loss_sc_norm_list = []
            for conv_name in stu_model.all_conv_names:
                if config['channel_select_algo'] == 'sparse_vec':
                    conv_cm_params = list(eval('stu_model.' + conv_name + '_cm').parameters())[0]
                    zero_target = Variable(torch.zeros(conv_cm_params.size()).cuda())
                    loss_reg_temp = criterion_l1(conv_cm_params, zero_target)
                    loss_reg_list.append(loss_reg_temp)
                elif config['channel_select_algo'] == 'subspace_cluster':
                    conv_layer_sc = eval('stu_model.' + conv_name + '_sc')
                    loss_recons = nn.MSELoss()(conv_layer_sc.diff_recons, conv_layer_sc.zeros_target_recons)
                    loss_norm = nn.L1Loss()(conv_layer_sc.fc1.weight, conv_layer_sc.zeros_target_norm)
                    loss_sc_recons_list.append(loss_recons)
                    loss_sc_norm_list.append(loss_norm)
                else:
                    raise ValueError
            if config['channel_select_algo'] == 'sparse_vec':
                pass
            elif config['channel_select_algo'] == 'subspace_cluster':
                loss_sc_recons = sum(loss_sc_recons_list)
                loss_sc_norm = sum(loss_sc_norm_list)
                loss = loss_sc_recons + config['gamma'] * loss_sc_norm

                print(
                    'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}\tLoss_R: {:.6f}\tLoss_N: {:.6f}\t'.format(
                        e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.data[0], loss_sc_recons.data[0], loss_sc_norm.data[0]
                    ))

        elif config['phase'] == 2:

            loss_A1_list = []
            loss_A2_list = []
            loss_fea_list = []

            for item in config['conv_pruned_names']:
                conv_name1, conv_name2, _ = item
                ''' 特征图 '''
                tea_conv1_fea_maps = tea_fea_maps[conv_name1]
                tea_conv2_fea_maps = tea_fea_maps[conv_name2]
                stu_conv1_fea_maps = stu_fea_maps[conv_name1]
                stu_conv2_fea_maps = stu_fea_maps[conv_name2]
                ''' attention map '''
                tea_A1 = att_map_layer(tea_conv1_fea_maps)
                tea_A2 = att_map_layer(tea_conv2_fea_maps)
                stu_A1 = att_map_layer(stu_conv1_fea_maps)
                stu_A2 = att_map_layer(stu_conv2_fea_maps)
                ''' attention_map 差异'''
                loss_A1_temp = criterion(stu_A1, tea_A1)
                loss_A2_temp = criterion(stu_A2, tea_A2)
                loss_A1_list.append(loss_A1_temp)
                loss_A2_list.append(loss_A2_temp)
                ''' 特征图差异 第二层卷积的输出'''
                loss_fea_temp = criterion(stu_conv2_fea_maps, tea_conv2_fea_maps)
                loss_fea_list.append(loss_fea_temp)

            loss_A = sum(loss_A1_list) + sum(loss_A2_list)
            loss_F = sum(loss_fea_list)
            loss = loss_F

            print(
                'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:15.6f}\tLoss_F: {:15.6f}\tLoss_A: {:15.12f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.data[0], loss_F.data[0], loss_A.data[0]
                ))

        elif config['phase'] == 3:
            loss = criterion_cls(stu_sample_scores,target)
            print(
                'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:15.6f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),loss.data[0]))

        ''' 4. 误差反向传播 '''
        loss.backward()
        ''' 5. 参数更新'''
        optimizer.step()

        epoch_loss.append(loss.data[0])
        """
        if config['phase'] == 1:
            if config['channel_select_algo'] == 'sparse_vec':
                '''
                计算sparse_vec中的最大值最小值
                '''
                one_cm_layer = eval('stu_model.'+config['conv_pruned_names'][0][0]+'_cm')
                sparse_min = list(one_cm_layer.parameters())[0].abs().min()
                sparse_max = list(one_cm_layer.parameters())[0].abs().max()

                print(
                'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}\tLoss_F: {:.6f}\tLoss_A: {:.12f}\tLoss_L1: {:.6f}\tmin: {:.6f}\tmax: {:.6f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.data[0],loss_F.data[0],loss_A.data[0], loss_reg.data[0], sparse_min.data[0], sparse_max.data[0]
                ))
        """
    print ('Epoch: {:3d}\taverage_epoch_loss: {:.6f}'.format(e,sum(epoch_loss)/len(epoch_loss)))

flag = True
for e in range(config['start_epoch'],config['epoch']+1):
    if config['phase'] == 1 and flag == True:
        test()
        flag = False
    elif config['phase'] == 2:
        test()
    train(e)
    if e % config['save_freq'] == 0 or e == config['epoch']:
        torch.save(stu_model.state_dict(),os.path.join(save_path,'stage{}_epoch{}.pth'.format(config['phase'],e)))
        pass
