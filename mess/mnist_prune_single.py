#coding=utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch import nn

from config import config
from mess.models import DemoNet
from data_loader import get_data_loader
from mine_layers import AttentionMap

# 加载数据集
train_loader, test_loader = get_data_loader("mnist")
# 加载teacher model
tea_model = DemoNet(weight_path=config['weight_path'])
# tea_weights = torch.load(config['weight_path'])
# tea_model.load_state_dict(tea_weights)
tea_model.cuda()
tea_model.train()
# 加载student model
stu_model = DemoNet(is_teacher=False,phase=config['phase'],weight_path=config['stu_weight_path'])
# stu_weights = stu_model.state_dict()
# if config['stu_weight_path'] == '':
#     stu_weights = inject_params(stu_weights,tea_weights) # student model 从 teacher model中加载参数
# else:
#     stu_weights = torch.load(config['stu_weight_path'])
# stu_model.load_state_dict(stu_weights)
stu_model.cuda()
stu_model.train()
# 仅仅优化student model需要优化的参数
# optimizer = optim.Adam(list(stu_model.conv1.parameters())+list(stu_model.cm1.parameters()),lr=config['lr'])
if config['phase'] == 1:
    optimizer = optim.Adam([
        {'params': stu_model.conv1.parameters()},
        # {'params': stu_model.cm1.parameters(),'weight_decay': config['gamma']} # 在优化器中加入L2 loss的方式
        {'params': stu_model.cm1.parameters()}
    ],lr=config['lr'])
elif config['phase'] == 2:
    optimizer = optim.Adam([
        {'params': stu_model.conv1.parameters()}
    ],lr=config['lr'])
else:
    raise ValueError,"config['phase'] = {}".format(config['phase'])
# 初始化工具层
att_map_layer = AttentionMap()
criterion = nn.MSELoss()
criterion_l1 = nn.L1Loss(size_average=False) # 使用L1 Loss来计算l1 范数

# 打印cm层学到的参数
# for param in stu_model.cm1.parameters():
#     print param.data

# 如果是阶段2，则进行剪枝
if config['phase'] == 2:
    # stu_model.prune()
    stu_model.cuda()


def test():
    stu_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = stu_model.forward(data,is_test=True)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for e in range(1,config['epoch']+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 只要有一个 input Variable是 volatile的，那么之后所有output的Variable都是volatile的，是无法进行求导的
        # tea_data, _ = Variable(data,volatile=True).cuda(), Variable(target).cuda() #本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        tea_data, _ = Variable(data.cuda(),volatile=True),Variable(target.cuda())#本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        stu_data, _ = Variable(data.cuda(),requires_grad=True),Variable(target.cuda())#本质上，不需要求关于input 以及 target的梯度，因为网络中参数的变化并不会导致input以及target的变化
        optimizer.zero_grad()
        tea_fea_maps = tea_model.forward(tea_data) # teacher model 仅仅需要 inference，不需要求梯度
        tea_fea_maps = Variable(tea_fea_maps.data.clone().cuda(),requires_grad=False)
        stu_fea_maps = stu_model.forward(stu_data)

        tea_A = att_map_layer(tea_fea_maps)
        stu_A = att_map_layer(stu_fea_maps)

        loss_A = criterion(stu_A,tea_A)
        if config['phase'] == 1:
            loss_l1 = []
            for param in stu_model.cm1.parameters():
                zero_target = Variable(torch.zeros(param.size())).cuda()
                loss_temp = criterion_l1(param,zero_target)
                loss_l1.append(loss_temp)
            loss_l1 = sum(loss_l1)
            loss_l1 = loss_l1 * config['gamma']
            loss = loss_A + loss_l1
        else:
            loss = loss_A
        loss.backward()
        #print stu_data.grad.data[0,0,0,0]
        optimizer.step()
        if config['phase'] == 1:
            """
            计算sparse_vec中的最大值最小值
            """
            sparse_min = list(stu_model.cm1.parameters())[0].abs().min()
            sparse_max = list(stu_model.cm1.parameters())[0].abs().max()

            print('Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}\tLoss_A: {:.6f}\tLoss_L1: {:.6f}\tmin: {:.6f}\tmax: {:.6f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0],
                loss_A.data[0],loss_l1.data[0],sparse_min.data[0],sparse_max.data[0]
            ))
        else:
            print(
            'Train Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]
            ))
    test()
    stu_model.train()
    if e % config['save_freq'] == 0:
        # torch.save(stu_model.state_dict(),'./weight/mnist-demonet/stage2_epoch{}.pth'.format(e))
        pass
