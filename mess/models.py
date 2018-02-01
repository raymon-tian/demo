#coding=utf-8
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
import numpy as np

from mine_layers import ChannelMultiplier
from mine_utils import get_top_k,inject_params

class DemoNet(nn.Module):
    """
    每个模型存在着两个阶段
    phase1 : 学习用于通道选择的稀疏向量，同时最小化Loss_Attention;i.e., min Loss_vec + Loss_A
    phase2 : 剪枝通道：从phase1阶段学习得到的稀疏向量中选取topk，然后剪枝
    """
    def __init__(self,is_teacher=True,phase=1,weight_path=None):
        """
        阶段1 ： 仅仅学习一个稀疏向量； phase = 2： 剪枝 + fine tune
        :param is_teacher: 是否为 teacher model
        :param phase: phase1：加载teacher_weight,每个conv层之后安装一个插件——CM层，为每一个conv层学习一个CM层，目标为 min知识误差 以及 稀疏化CM层
                      phase2：加载p1_weight，prune，之后min知识误差，最后模型存储的时候，去掉对应的CM层
        :param weight_path: 权重路径
        """
        super(DemoNet, self).__init__()
        self.phase = phase
        self.is_teacher = is_teacher
        self.weight_path = weight_path
        ''' 初始化对应的网络层'''
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #(1,10,5,5)
        if (not is_teacher) and phase == 1:
            self.cm1 = ChannelMultiplier(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        '''加载权重'''
        if self.weight_path is not None:
            refer_weights = torch.load(self.weight_path)
            own_weights = self.state_dict()
            if self.is_teacher:
                self.load_state_dict(refer_weights)
            elif self.phase == 1:
                own_weights = inject_params(own_weights,refer_weights)
                self.load_state_dict(own_weights)
            else:
                # 如果初始参数中存在CM层，则表明phase2阶段加载的weights来自于phase1，需要1.初始化一个CM层,加载参数 2.prune
                need_prune = False
                for k in refer_weights.keys():
                    if 'multiplier' in k:
                        need_prune = True
                        break
                if need_prune:
                    self.cm1 = ChannelMultiplier(10) # 添加对应的缺少的CM层
                    self.compress_ratio = 9
                    own_weights = self.state_dict()
                    own_weights = inject_params(own_weights,refer_weights) # 拷贝相应的参数
                    self.load_state_dict(own_weights)
                    self.__prune() # 进行剪枝
                else:
                    self.load_state_dict(refer_weights)
        # self.__reinit_layer('conv1',**{})
        # self.__reinit_layer('conv2',**{})

    def forward(self, x,is_test=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 剪枝哪一层就在哪一层后面加入CM层插件
        if (not self.is_teacher) and (is_test is False):
            if self.phase == 1:
                x_weighted = self.cm1(x)
                return x_weighted
            else:
                return x
        elif is_test is False:
            # 到此处，必然 is_teacher = True
            return x
        else:
            # 到此处，is_test 为 True; is_teacher情况未知
            if self.is_teacher is not True:
                if self.phase == 1:
                    x = self.cm1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def __prune(self):
        """
        根据学习得到的权重向量剪枝，剪枝对应的 i-th层 以及 (i+1)-th 层的conv层;
        :return:
        """
        # 在这一层self.is_teacher要设置为True，因为训练的时候不再需要CM层
        # self.is_teacher = True
        weight_vec = self.cm1.multiplier.data.cpu().numpy()
        # 必须对weight_vec取绝对值，否则就是bug
        weight_vec = np.abs(weight_vec)
        # edit! 设置超参数:压缩率
        _, indices = get_top_k(weight_vec,self.compress_ratio)
        indices = sorted(indices)
        # 剪枝 i-th层，剪枝0-th axis ： 1.重新初始化对应的层 2.拷贝参数
        ori_weight = self.conv1.weight.data.clone()
        ori_bias = self.conv1.bias.data.clone()
        self.__reinit_layer(name='conv1',out_channels=len(indices))
        self.conv1.weight.data = ori_weight[indices,:,:,:]
        self.conv1.bias.data = ori_bias[indices,]
        # 剪枝 i+1-th层，剪枝1-th axis ： 1.重新初始化对应的层 2.拷贝参数
        ori_weight = self.conv2.weight.data.clone()
        self.__reinit_layer(name='conv2', in_channels=len(indices))
        self.conv2.weight.data = ori_weight[:, indices, :, :]

    def __inject_pruned_params(self,path):
        """
        将剪枝后的参数加载到网络中去，这个必须要在构造函数中初始化各个子层的之后调用
        :param path:
        :return:
        """
        weights_pruned = torch.load(path)
        keys = weights_pruned.keys()
        num_key = len(keys)
        i = 0
        while i < num_key:
            # check，确保key出现的形式如'conv1.weight'
            assert len(keys[i].split('.')) == 2, "模型参数的名称不是形如'conv1.weight'的形式"
            layer_name = keys[i].split('.')[0]
            # 判断层的类型是否为卷积层
            layer = eval('self.'+layer_name)
            if type(layer) is nn.Conv2d:
                ori_shape = np.array(layer.weight.size())
                refer_shape = np.array(weights_pruned[keys[i]].size())
                flag = np.logical_and.reduce(ori_shape == refer_shape)
                # 如果shape不一致，则重新初始化该层,先不进行参数的拷贝，到最后一次性进行拷贝
                if not flag:
                    in_channels = refer_shape[1]
                    out_channels = refer_shape[0]
                    kernel_size = (refer_shape[2],refer_shape[3])
                    stride = layer.stride
                    padding = layer.padding
                    dilation = layer.dilation
                    groups = layer.groups
                    bias = layer.bias
                    setattr(self,layer_name,nn.Conv2d(
                        in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias
                    ))
                i += 1 # 跳过下面的bias参数
        # 最后进行参数的拷贝
        self.load_state_dict(weights_pruned)

    def __reinit_layer(self,name,**kwargs):
        """
        该函数只能在构造函数内调用!
        根据name，重新初始化某一层
        :param name:
        :param kwargs:
        :return:
        """
        layer = eval('self.'+name)
        # 原始参数
        ori_params = dict()
        ori_params['in_channels'] = layer.in_channels
        ori_params['out_channels'] = layer.out_channels
        ori_params['kernel_size'] = layer.kernel_size
        ori_params['stride'] = layer.stride
        ori_params['padding'] = layer.padding
        ori_params['dilation'] = layer.dilation
        ori_params['groups'] = layer.groups
        ori_params['bias'] = layer.bias is not None
        # 更新参数
        for k,v in kwargs.items():
            ori_params[k] = v
        delattr(self,name)
        setattr(self,name,nn.Conv2d(**ori_params))
        """
        setattr(self,name,nn.Conv2d(
            in_channels=ori_params['in_channels'],
            out_channels=ori_params['out_channels'],
            kernel_size=ori_params['kernel_size'],
            stride=ori_params['stride'],
            padding=ori_params['padding'],
            dilation=ori_params['dilation'],
            groups=ori_params['groups'],
            bias=ori_params['bias']
        ))
        """

if __name__ == '__main__':
    demo = DemoNet()
