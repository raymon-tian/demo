#coding=utf-8
from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms,models

from config import config
from mine_folder import ImageFolder

cuda = config['cuda']
seed = config['seed']
batch_size = config['batch_size']
test_batch_size = config['test_batch_size']
topC = config['topC']
randomN = config['randomN']

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

def get_data_loader(name=None):
    """

    :param name:
    :return:
    """
    if name is None:
        name = config['dataset_name']
    if name == "mnist":
        return mnist_data_loader()
    elif name == 'imagenet12':
        return imagenet12_data_loader()
    elif name == "cifar10":
        return cifar10_data_loader()
    elif name == "cifar100":
        pass
    elif name == "cifar1000":
        pass
    else:
        pass

def mnist_data_loader():
    train_loader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader,test_loader

def cifar10_data_loader():
    cifar10_train_dataset = datasets.CIFAR10(root='./data/cifar',train=True,download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                        ]))
    cifar10_test_dataset = datasets.CIFAR10(
        root='./data/cifar', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]))
    train_loader = DataLoader(cifar10_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    test_loader = DataLoader(cifar10_test_dataset,batch_size=test_batch_size,shuffle=True,**kwargs)

    return train_loader,test_loader

def imagenet12_data_loader():

    traindir = './data/imagenet12/train'
    valdir = './data/imagenet12/val'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder(
        root=traindir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        topC=0,
        randomN=randomN
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,**kwargs)

    val_dataset = ImageFolder(
        root=valdir,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        topC=topC,
        randomN=0
    )
    val_loader = DataLoader(val_dataset,batch_size=test_batch_size,shuffle=False,**kwargs)

    return train_loader,val_loader

if __name__ == '__main__':
    train_loader, val_dataset = imagenet12_data_loader()
    for batch_idx, (data, target) in enumerate(train_loader):
        pass