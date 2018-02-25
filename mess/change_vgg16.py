#coding=utf-8

import torch
from collections import OrderedDict

mine_map = {
    0: 'conv1_1',
    2: 'conv1_2',
    5: 'conv2_1',
    7: 'conv2_2',
    10:'conv3_1',
    12:'conv3_2',
    14:'conv3_3',
    17:'conv4_1',
    19:'conv4_2',
    21:'conv4_3',
    24:'conv5_1',
    26:'conv5_2',
    28:'conv5_3'
}

def get_name(n1):
    ns = n1.split('.')
    if ns[0] == 'classifier':
        return  n1
    if ns[2] == 'weight':
        temp = mine_map[int(ns[1])]+'.weight'
        return temp
    else:
        temp = mine_map[int(ns[1])] + '.bias'
        return temp



def change_names():
    path = '../pretrained/vgg16.pth'
    w = torch.load(path)
    d2 = OrderedDict([(get_name(k), v) for k, v in w.items()])
    torch.save(d2,'/home/raymon/workspace/prune_kd/weight/imagenet12-vgg16/vgg16-org.pth')


if __name__ == '__main__':
    change_names()