#coding=utf-8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # 就是先调用了父类的构造函数一下
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.register_parameter('fc1_weight', None)
        self.register_parameter('fc1_bias', None)
        self.register_parameter('fc2_weight', None)
        self.register_parameter('fc2_bias', None)
        self.fc1_weight = nn.Parameter(torch.randn(50, 320).cuda())
        # self.fc1_bias = self.fc1.bias
        self.fc1_bias = nn.Parameter(torch.randn(50).cuda())
        self.fc2_weight = nn.Parameter(torch.randn(10, 50).cuda())
        self.fc2_bias = nn.Parameter(torch.randn(10).cuda())
    def forward(self, x):
        # if hasattr(self,'conv1') == False:
        #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #     self.add_module('conv1',nn.Conv2d(1, 10, kernel_size=5))
            # self.register_parameter('conv1.weight',self.conv1.weight)
            # self.register_parameter('conv1.bias',self.conv1.bias)
        # if hasattr(self,'conv2') == False:
        #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # if hasattr(self,'conv2_drop') == False:
        #     self.conv2_drop = nn.Dropout2d()
        #
        # if hasattr(self,'fc1') == False:
        if self.fc1_weight is None:
            # nn.Linear
            # self.add_module('fc1',nn.Linear(320,50).cuda())
            # self.fc1_weight = self.fc1.weight
            # self.fc1_weight @ x
            self.fc1_weight = nn.Parameter(torch.randn(50,320).cuda())
            # self.fc1_bias = self.fc1.bias
            self.fc1_bias = nn.Parameter(torch.randn(50).cuda())
        else:
            pass
            # self.fc1.weight = self.fc1_weight
            # self.fc1.bias = self.fc1_bias
            # self.register_parameter('fc1.weight', self.fc1.weight)
        # if hasattr(self,'fc2') == False:
        if self.fc2_weight is None:
            self.fc2_weight = nn.Parameter(torch.randn(10,50).cuda())
            self.fc2_bias = nn.Parameter(torch.randn(10).cuda())
            # self.add_module('fc2',nn.Linear(50,10).cuda())
            # self.register_parameter('fc2.weight', self.fc2.weight)
            # self.fc2_weight = self.fc2.weight
            # self.fc2_biase = self.fc2.bias
        else:
            pass
            # self.fc2.weight = self.fc2_weight
            # self.fc2.bias = self.fc2_bias
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        x = F.relu(F.linear(x,self.fc1_weight,self.fc1_bias))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x = F.linear(x,self.fc2_weight,self.fc2_bias)
        return F.log_softmax(x)

model = Net()
# if args.cuda:
#     model.cuda()
ps = model.parameters()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    # torch.save(model.state_dict(),'mnist_org.ckpt')
    test()