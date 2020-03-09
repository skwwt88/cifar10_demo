import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.optim import lr_scheduler
from collections import namedtuple, defaultdict

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
]

batch_size = 512
transform = transforms.Compose(
    [transforms.Pad(2),
     transforms.RandomCrop(32),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0, 0, 0),  (1, 1, 1)),
     transforms.RandomErasing(0.9, scale=(0.01, 0.03))])

transform = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomAffine(30),
     transforms.ToTensor(),
     transforms.RandomErasing(0.9, scale=(0.04, 0.08))])

test_transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Module):
    def __init__(self, c_in, c_out, k_size = 3, padding = 1):
        super().__init__()
        self.conv1 =  nn.Conv2d(c_in, c_out, kernel_size=k_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu1 = nn.ReLU(True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class Residual(nn.Module):
    def __init__(self, channels, k_size = 3):
        super().__init__()
        self.convbn1 =  ConvBn(channels, channels, k_size=k_size)
        self.convbn2 =  ConvBn(channels, channels, k_size=k_size)
        
    def forward(self, input):
        x = self.convbn1(input)
        x = self.convbn2(x)
        x = x.add(input)
        return x

class InceptionModule(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.branch1 = ConvBn(c_in, c_out, 3)
        self.branch2 = ConvBn(c_in, c_out, 5, padding=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        return out

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prepare = ConvBn(3, 64) 
        self.pool = nn.MaxPool2d(2)
        self.conv_layer1 = ConvBn(64, 128)
        self.res_layer1 = Residual(128)
        self.conv_layer2 = ConvBn(128, 256)
        self.res_layer2 = Residual(256)
        self.conv_layer3 = ConvBn(256, 512)
        self.res_layer3 = Residual(512)
        self.max_pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10, bias=False)
        self.res4 = Residual(1024)
        self.pool2 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(4)
        self.dropout = nn.Dropout2d(0.2)
        self.mul = Mul(0.125)

    def forward(self, x):
        x = self.prepare(x)
        x = self.pool(self.conv_layer1(x))
        x = self.res_layer1(x)
        x = self.pool(self.conv_layer2(x))
        x = self.res_layer2(x)
        x = self.pool(self.conv_layer3(x))
        x = self.res_layer3(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), x.size(1))
        # x = nn.Dropout(0.05)(x)
        x = self.fc(x)

        return x

net = Net().to(device=device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss(reduction='none')


#scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

lr_schedule = PiecewiseLinear([0, 30, 100], [0, 6, 0])

optimizer = optim.SGD(net.parameters(), lr=0.000009, momentum=0.9, weight_decay=5e-4 * batch_size, nesterov=True)



for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    tcorrect = 0
    ttotal = 0

    w = 1.0
    lr = lambda step: lr_schedule(step) * w
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        
        _, tpredicted = torch.max(outputs.data, 1)
        ttotal += labels.size(0)
        tcorrect += (tpredicted == labels.to(device)).sum().item()
    
    train_acc = 100 * tcorrect / ttotal

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print('epoch {0} train_acc: {1} valid_acc: {2}'.format(epoch, train_acc, (100 * correct / total)))
print('Finished Training')

