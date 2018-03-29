#coding=utf-8
import torch
from torch.autograd import Variable
from torch import nn
from collections import OrderedDict

from resnet import ResNet50
from config import config

resnet50_conv_names = OrderedDict({

    'conv1' : None,
    'layer1_block0_conv1' : None,
    'layer1_block0_conv2' : None,
    'layer1_block0_conv3' : None,
    'layer1_block0_shortcut_conv' : None,
    'layer1_block1_conv1' : None,
    'layer1_block1_conv2' : None,
    'layer1_block1_conv3' : None,
    'layer1_block2_conv1' : None,
    'layer1_block2_conv2' : None,
    'layer1_block2_conv3' : None,
    'layer2_block0_conv1' : None,
    'layer2_block0_conv2' : None,
    'layer2_block0_conv3' : None,
    'layer2_block0_shortcut_conv' : None,
    'layer2_block1_conv1' : None,
    'layer2_block1_conv2' : None,
    'layer2_block1_conv3' : None,
    'layer2_block2_conv1' : None,
    'layer2_block2_conv2' : None,
    'layer2_block2_conv3' : None,
    'layer2_block3_conv1' : None,
    'layer2_block3_conv2' : None,
    'layer2_block3_conv3' : None,
    'layer3_block0_conv1' : None,
    'layer3_block0_conv2' : None,
    'layer3_block0_conv3' : None,
    'layer3_block0_shortcut_conv' : None,
    'layer3_block1_conv1' : None,
    'layer3_block1_conv2' : None,
    'layer3_block1_conv3' : None,
    'layer3_block2_conv1' : None,
    'layer3_block2_conv2' : None,
    'layer3_block2_conv3' : None,
    'layer3_block3_conv1' : None,
    'layer3_block3_conv2' : None,
    'layer3_block3_conv3' : None,
    'layer3_block4_conv1' : None,
    'layer3_block4_conv2' : None,
    'layer3_block4_conv3' : None,
    'layer3_block5_conv1' : None,
    'layer3_block5_conv2' : None,
    'layer3_block5_conv3' : None,
    'layer4_block0_conv1' : None,
    'layer4_block0_conv2' : None,
    'layer4_block0_conv3' : None,
    'layer4_block0_shortcut_conv' : None,
    'layer4_block1_conv1' : None,
    'layer4_block1_conv2' : None,
    'layer4_block1_conv3' : None,
    'layer4_block2_conv1' : None,
    'layer4_block2_conv2' : None,
    'layer4_block2_conv3' : None,

})

def get_resnet50_conv_names():
    model = ResNet50()
    added_names = []

    for k in model.state_dict().keys():
        k_splits = k.split('.')
        del k_splits[-1]
        if len(k_splits) == 1:
            layer_name = '.'.join(k_splits)
        elif len(k_splits) == 3:
            k_splits[0] += '['+k_splits[1]+']'
            del k_splits[1]
            layer_name = '.'.join(k_splits)
        elif len(k_splits) == 4:
            k_splits[0] += '[' + k_splits[1] + ']'
            k_splits[-2] += '[' + k_splits[-1] + ']'
            del k_splits[1]
            del k_splits[-1]
            layer_name = '.'.join(k_splits)
        else:
            raise ValueError

        if(isinstance(eval('model.'+layer_name),nn.Conv2d)):
            if (layer_name in added_names) is False:
                added_names.append(layer_name)
                print("'"+layer_name+"'")
    return added_names

def get_conv_names():
    if config['model_name'] == 'vgg16':
        pass
    elif config['model_name'] == 'resnet50':
        return resnet50_conv_names
    else:
        raise ValueError

if __name__ == '__main__':
    get_resnet50_conv_names()
