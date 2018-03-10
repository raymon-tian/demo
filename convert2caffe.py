#coding=utf-8
import numpy as np
import sys
caffe_path = '/home/raymon/caffe/python'
sys.path.insert(0,caffe_path)
import caffe

model_path_torch = '/home/raymon/workspace/demo/weight/imagenet12-vgg16/random/vis_fea/conv1_1/x8/stage2_epoch30.pth'
model_path_caffe = '/home/raymon/workspace/demo/weight/caffe/vgg16.caffemodel'
prototxt_path = '/home/raymon/workspace/demo/weight/caffe/vgg16-deploy.prototxt'


# caffe.set_mode_cpu()
net = caffe.Net(prototxt_path, caffe.TEST)

import torch
weight_t = torch.load(model_path_torch)
# print(weight_t.keys())

# %paste
for k in weight_t.keys():
    k_w = k.split('.')
    if len(k_w) == 2 and ('conv' in k_w[0]):
        print(k)
        if k_w[1] == 'weight':
            t1 = net.params[k_w[0]][0].data
            t2 = weight_t[k].cpu().numpy()
            assert t1.shape == t2.shape
            net.params[k_w[0]][0].data[...] = weight_t[k].cpu().numpy()
        elif k_w[1] == 'bias':
            net.params[k_w[0]][1].data[...] = weight_t[k].cpu().numpy()
        else:
            raise ValueError

net.save(model_path_caffe)
