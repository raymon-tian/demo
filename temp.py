#coding=utf-8
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import models

"""
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.multp = Variable(torch.rand(1), requires_grad=True)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.multp = nn.Parameter(torch.rand(1)) # requires_grad is True by default for Parameter

m1 = Model1()
m2 = Model2()

p = m2.parameters()
print('m1', list(m1.parameters()))
print('m2', list(m2.parameters()))
print(m2.parameters()[0])
"""

class Test(nn.Module):
    def __init__(self,i):
        super(Test,self).__init__()
        self.p = nn.Parameter(torch.from_numpy([i]))
    def forward(self, x):
        return x * self.p

####
# 所以证明了 pytorch 会将叶子节点存储下来
"""
if __name__ == '__main__':
    x = Variable(torch.Tensor([2]),requires_grad = True) # x=2
    flag = False
    for i in range(2,5):
        mask = Variable(torch.Tensor([i]))
        if flag == False:
            y = x * mask
            flag = True
        else:
            y = y * mask
    print x.grad None
    y.backward()
    # print y.grad # None
    print x.grad # None 
"""

def a_test():
    print('aaa')

if __name__ == '__main__':
    vgg16 = models.vgg16().cuda()
    x = Variable(torch.randn((1,3,224,224).cuda()))
    vgg16.forward(x)
    x = Variable(torch.randn((2,4,12*12)).cuda())
    L = nn.Linear(12*12,16).cuda()
    y = L(x)
    x = Variable(torch.Tensor([2]),requires_grad = True) # x=2
    t = x
    for i in range(2,5):
        mask = Variable(torch.Tensor([i]))
        x = x * mask
    print x.grad # None
    x.backward()
    # print x.grad # None
    print t.grad # None
    eval('a_test')()