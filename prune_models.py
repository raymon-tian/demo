#coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from mine_layers import ChannelMultiplier,SubspaceCluster
from mine_utils import get_top_k,inject_params,spectrum_cluster

class ChannelPruneNet(nn.Module):
    """
    每个模型存在着两个阶段
    phase1 : 学习用于通道选择的稀疏向量，同时最小化Loss_Attention;i.e., min Loss_vec + Loss_A
    phase2 : 剪枝通道：从phase1阶段学习得到的稀疏向量中选取topk，然后剪枝
    """
    def __init__(self,model_name,channel_select_algo=None,is_teacher=True,phase=1,weight_path=None,conv_names=None,num_classes=1000,fine_tune=False,**kwargs):
        """
        阶段1 ： 仅仅学习一个稀疏向量； phase = 2： 剪枝 + fine tune
        :param model_name: 待剪枝的模型的名称
        :param is_teacher: 是否为 teacher model
        :param phase: phase1：加载teacher_weight,每个conv层之后安装一个插件——CM层，为每一个conv层学习一个CM层，目标为 min知识误差 以及 稀疏化CM层
                      phase2：加载p1_weight，prune，之后min知识误差，最后模型存储的时候，去掉对应的CM层
        :param weight_path: 权重路径
        :param conv_names ((e1,e2,ratio),...) 表示前一层，后一层conv层的名字，以及对前一层的压缩率
        """
        assert model_name in ['vgg16','resNet50','demo'], "不支持的网络模型"
        assert channel_select_algo in ['subspace_cluster','random','thinet','sparse_vec',None],"不支持的通道选择算法"
        assert (weight_path is not None) and (weight_path is not ''),"weight_path 非法"
        assert phase == 1 or phase == 2, "phase非法"
        super(ChannelPruneNet, self).__init__()
        self.model_name = model_name
        self.channel_select_algo = channel_select_algo
        self.phase = phase
        self.is_teacher = is_teacher
        self.weight_path = weight_path
        self.conv_names = conv_names
        self.num_classes = num_classes
        self.fine_tune = fine_tune
        ''' 存储sc算法产生的loss'''
        # self.loss_sc_recons_list = []
        # self.loss_sc_norm_list = []
        # self.loss_sc_inrecons_list = []
        ''' 初始化原始网络对应的网络层'''
        getattr(self,model_name+'_init')()
        ''' 安装插件：初始化 待压缩的卷积层的 CM层插件 或者 subspace层 插件 该阶段不需要压缩率信息 以及 第二层的名字'''
        # if (not is_teacher) and phase == 1:
        if not is_teacher:
            self.is_init_forward = True
            ''' 固定网络的输入，进行demo演示，获得各个卷积模块的特征图'''
            if model_name == 'demo':
                self.demo_input = Variable(torch.randn(1, 1, 28, 28))
            elif model_name == 'vgg16':
                self.demo_input = Variable(torch.randn(1, 3, 224, 224))
            _, demo_output = getattr(self, model_name + '_forward')(self.demo_input)
            self.is_init_forward = False
            for item in conv_names:
                conv_n1, _, _ = item
                out_channels = eval('self.'+conv_n1+'.out_channels')
                setattr(self,conv_n1+'_out_channels',out_channels)
                if self.channel_select_algo == 'sparse_vec':
                    setattr(self,conv_n1+'_cm',ChannelMultiplier(out_channels))
                elif self.channel_select_algo == 'subspace_cluster':
                    _,fea_C,fea_H,fea_W = demo_output[conv_n1].size()
                    # edit! 超参数
                    setattr(self,conv_n1+'_sc',SubspaceCluster(H=fea_H,W=fea_W,K=fea_C))
        ''' 加载权重 '''
        if self.weight_path is not None:
            refer_weights = torch.load(self.weight_path)
            own_weights = self.state_dict()
            if self.is_teacher:
                self.load_state_dict(refer_weights)
            elif self.phase == 1:
                own_weights = inject_params(own_weights,refer_weights)
                self.load_state_dict(own_weights)
            else:
                """
                # 如果初始参数中存在CM层，则表明phase2阶段加载的weights来自于phase1，需要1.初始化一个CM层,加载参数 2.prune
                need_prune = False
                for k in refer_weights.keys():
                    # bug when 参数中有 multiplier，但是不是CM层
                    if 'cm.multiplier' in k:
                        need_prune = True
                        break
                if need_prune:
                    # 添加对应的缺少的CM层
                    for item in conv_names:
                        conv_n1, _, compress_ratio = item
                        out_channels = eval('self.' + conv_n1 + '.out_channels')
                        setattr(self, conv_n1 + '_out_channels', out_channels)
                        setattr(self, conv_n1 + '_cm', ChannelMultiplier(out_channels))
                        setattr(self, conv_n1 + '_ratio', compress_ratio)
                    own_weights = self.state_dict()
                    own_weights = inject_params(own_weights,refer_weights) # 拷贝相应的参数
                    self.load_state_dict(own_weights)
                    self.__prune() # 进行剪枝
                else:
                    self.load_state_dict(refer_weights)
                """
                if self.fine_tune:
                    # self.__prune()
                    self.load_state_dict(refer_weights)
                else:
                    self.load_state_dict(refer_weights)
                    self.__prune()

    def vgg16_init(self):
        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            # 输入大小固定，必须是(N,3,224,224)
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def vgg16_forward(self,x):
        out_fea_maps = dict()

        conv1_1_x = self.conv1_1(x)
        conv1_1_x = self.exec_plugin('conv1_1',conv1_1_x)
        out_fea_maps['conv1_1'] = conv1_1_x
        conv1_1_x = self.relu1_1(conv1_1_x)

        conv1_2_x = self.conv1_2(conv1_1_x)
        conv1_2_x = self.exec_plugin('conv1_2', conv1_2_x)
        out_fea_maps['conv1_2'] = conv1_2_x
        conv1_2_x = self.relu1_2(conv1_2_x)
        conv1_p = self.pool1(conv1_2_x)

        conv2_1_x = self.conv2_1(conv1_p)
        conv2_1_x = self.exec_plugin('conv2_1',conv2_1_x)
        out_fea_maps['conv2_1'] = conv2_1_x
        conv2_1_x = self.relu2_1(conv2_1_x)

        conv2_2_x = self.conv2_2(conv2_1_x)
        conv2_2_x = self.exec_plugin('conv2_2',conv2_2_x)
        out_fea_maps['conv2_2'] = conv2_2_x
        conv2_2_x = self.relu2_2(conv2_2_x)
        conv2_p = self.pool2(conv2_2_x)

        conv3_1_x = self.conv3_1(conv2_p)
        conv3_1_x = self.exec_plugin('conv3_1',conv3_1_x)
        out_fea_maps['conv3_1'] = conv3_1_x
        conv3_1_x = self.relu3_1(conv3_1_x)

        conv3_2_x = self.conv3_2(conv3_1_x)
        conv3_2_x = self.exec_plugin('conv3_2',conv3_2_x)
        out_fea_maps['conv3_2'] = conv3_2_x
        conv3_2_x = self.relu3_2(conv3_2_x)

        conv3_3_x = self.conv3_3(conv3_2_x)
        conv3_3_x = self.exec_plugin('conv3_3',conv3_3_x)
        out_fea_maps['conv3_3'] = conv3_3_x
        conv3_3_x = self.relu3_3(conv3_3_x)
        conv3_p = self.pool3(conv3_3_x)

        conv4_1_x = self.conv4_1(conv3_p)
        conv4_1_x = self.exec_plugin('conv4_1',conv4_1_x)
        out_fea_maps['conv4_1'] = conv4_1_x
        conv4_1_x = self.relu4_1(conv4_1_x)

        conv4_2_x = self.conv4_2(conv4_1_x)
        conv4_2_x = self.exec_plugin('conv4_2',conv4_2_x)
        out_fea_maps['conv4_2'] = conv4_2_x
        conv4_2_x = self.relu4_2(conv4_2_x)

        conv4_3_x = self.conv4_3(conv4_2_x)
        conv4_3_x = self.exec_plugin('conv4_3',conv4_3_x)
        out_fea_maps['conv4_3'] = conv4_3_x
        conv4_3_x = self.relu4_3(conv4_3_x)
        conv4_p = self.pool3(conv4_3_x)

        conv5_1_x = self.conv5_1(conv4_p)
        conv5_1_x = self.exec_plugin('conv5_1',conv5_1_x)
        out_fea_maps['conv5_1'] = conv5_1_x
        conv5_1_x = self.relu5_1(conv5_1_x)

        conv5_2_x = self.conv5_2(conv5_1_x)
        conv5_2_x = self.exec_plugin('conv5_2',conv5_2_x)
        out_fea_maps['conv5_2'] = conv5_2_x
        conv5_2_x = self.relu5_2(conv5_2_x)

        conv5_3_x = self.conv5_3(conv5_2_x)
        conv5_3_x = self.exec_plugin('conv5_3',conv5_3_x)
        out_fea_maps['conv5_3'] = conv5_3_x
        conv5_3_x = self.relu5_3(conv5_3_x)

        conv5_p = self.pool3(conv5_3_x)

        conv5_p = conv5_p.view(conv5_p.size(0), -1)

        y_p = self.classifier(conv5_p)

        return F.log_softmax(y_p),out_fea_maps

    def demo_init(self):
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #(1,10,5,5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def demo_forward(self,x):

        conv_output = dict()

        x1 = self.conv1(x)
        x1 = self.exec_plugin('conv1',x1)
        conv_output['conv1'] = x1
        x1 = F.relu(F.max_pool2d(x1, 2))

        x2 = self.conv2(x1)
        x2 = self.exec_plugin('conv2',x2)
        conv_output['conv2'] = x2
        x2 = F.relu(F.max_pool2d(self.conv2_drop(x2), 2))

        x = x2.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x),conv_output

    def exec_plugin(self,name,x,):
        """"""
        if self.is_teacher or self.is_init_forward or self.training == False:
            return x
        flag = False
        for item in self.conv_names:
                conv_n1, _, _ = item
                if name == conv_n1:
                    flag = True
                    break
        if flag is False or self.phase == 2 or self.is_teacher == True:
            return x
        else:
            if self.channel_select_algo == 'sparse_vec':
                return eval('self.'+name+'_cm')(x)
            elif self.channel_select_algo == 'subspace_cluster':
                x = eval('self.'+name+'_sc')(x)
                # self.loss_sc_recons_list.append(the_loss_sc[0])
                # self.loss_sc_inrecons_list.append(the_loss_sc[1])
                # self.loss_sc_norm_list.append(the_loss_sc[2])
                return x

    def forward(self,x,is_test=False):
        # self.loss_sc_norm_list[:] = []
        # self.loss_sc_inrecons_list[:] = []
        # self.loss_sc_recons_list[:] = []
        y_p, conv_output = getattr(self,self.model_name+'_forward')(x)
        return y_p, conv_output

    def __prune(self):
        """
        根据学习得到的权重向量剪枝，剪枝对应的 i-th层 以及 (i+1)-th 层的conv层;
        :return:
        """
        for item in self.conv_names:
            conv_n1,conv_n2,ratio = item

            if self.channel_select_algo == 'subspace_cluster':
                self.__subspace_cluster_prune(conv_n1,conv_n2,ratio)
            elif self.channel_select_algo == 'sparse_vec':
                self.__sparse_vec_prune(conv_n1,conv_n2,ratio)
            else:
                pass

    def __sparse_vec_prune(self,conv_n1,conv_n2,ratio):
        """
        
        :param conv_n1: 
        :param conv_n2: 
        :param ratio: 
        :return: 
        """''' 获取要保留的索引 '''
        weight_vec = eval('self.'+conv_n1+'_cm').multiplier.data.cpu().numpy()
        # 必须对weight_vec取绝对值，否则就是bug
        weight_vec = np.abs(weight_vec)
        left_out_channels = int(np.ceil(eval('self.'+conv_n1+'_out_channels') * ratio))
        assert left_out_channels != 0,"left_out_channels 不能为0"
        left_weight, indices = get_top_k(weight_vec,left_out_channels)
        sorted_indices = sorted(range(len(indices)), key=lambda k: indices[k])
        left_weight = [left_weight[sorted_indices[i]] for i in range(left_out_channels)]
        indices = sorted(indices)
        # 剪枝 i-th层，剪枝0-th axis ： 1.重新初始化对应的层 2.拷贝参数
        conv_layer1 = eval('self.'+conv_n1)
        conv_layer2 = eval('self.'+conv_n2)
        ori_weight = conv_layer1.weight.data.clone()
        ori_bias = conv_layer1.bias.data.clone()
        self.__reinit_layer(name=conv_n1,out_channels=len(indices))
        conv_layer1.weight.data = ori_weight[indices,:,:,:]
        conv_layer1.bias.data = ori_bias[indices,]
        ''' 对第i层的参数进行融合 '''
        for c in range(left_out_channels):
            conv_layer1.weight.data[c,:,:,:] *= float(left_weight[c])
            conv_layer1.bias.data[c] *= float(left_weight[c])
            pass
        # 剪枝 i+1-th层，剪枝1-th axis ： 1.重新初始化对应的层 2.拷贝参数
        ori_weight = conv_layer2.weight.data.clone()
        self.__reinit_layer(name=conv_n2,in_channels=len(indices))
        conv_layer2.weight.data = ori_weight[:, indices, :, :]

    def __subspace_cluster_prune(self,conv_n1,conv_n2,ratio):
        """
        使用subspace聚类算法剪枝卷积层
        :param conv_n1: str
        :param conv_n2: str
        :param ratio: float
        :return:
        """

        self_repres_matrix = eval('self.'+conv_n1+'_sc').fc1.weight.data.cpu().numpy()
        left_out_channels = int(np.ceil(eval('self.'+conv_n1+'_out_channels') * ratio))
        assert left_out_channels != 0,"left_out_channels 不能为0"
        cluster_result = spectrum_cluster(self_repres_matrix,left_out_channels)
        cluster_kernel_w_result = []
        cluster_kernel_b_result = []
        ''' 聚类第一层卷积kernel的weight以及bias '''
        conv_layer1 = eval('self.' + conv_n1)
        conv1_ori_weight = conv_layer1.weight.data.clone()
        conv1_ori_bias = conv_layer1.bias.data.clone()
        for i in range(len(cluster_result)):
            cluster_k_w = conv1_ori_weight[cluster_result[i],:,:,:]
            cluster_mean_k_w = torch.mean(source=cluster_k_w,dim=0,keepdim=True)
            cluster_mean_k_b = torch.mean(source=conv1_ori_bias[cluster_result[i]])
            cluster_kernel_w_result.append(cluster_mean_k_w)
            cluster_kernel_b_result.append(torch.Tensor([cluster_mean_k_b]))
        conv1_result_w = torch.cat(cluster_kernel_w_result)
        conv1_result_b = torch.cat(cluster_kernel_b_result)
        ''' 聚类第二层卷积kernel '''
        conv_layer2 = eval('self.' + conv_n2)
        conv2_result_w = []
        n_output = conv_layer2.weight.size()[0]
        for i in range(n_output):
            temp = []
            single_k = conv_layer2.weight.data[i]
            for j in range(len(cluster_result)):
                temp.append(torch.mean(single_k[cluster_result[j],:,:],dim=0,keepdim=True))
            conv2_result_w.append(torch.cat(temp).unsqueeze(0))
        conv2_result_w = torch.cat(conv2_result_w)
        ''' 重新初始化第一层卷积'''
        self.__reinit_layer(name=conv_n1, out_channels=len(cluster_result))
        conv_layer1.weight.data = conv1_result_w
        conv_layer1.bias.data = conv1_result_b
        ''' 重新初始化第二层卷积'''
        self.__reinit_layer(name=conv_n2,in_channels=len(cluster_result))
        conv_layer2.weight.data = conv2_result_w

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

    def remove_cm(self):
        """
        在存储模型之前去除掉 CM 层
        :return:
        """
        assert self.phase == 2, "只能去除第二阶段的CM层"
        for item in self.conv_names:
            conv_n1,_,_ = item
            delattr(self,conv_n1+'_cm')
if __name__ == '__main__':
    # demo = DemoNet()
    pass
