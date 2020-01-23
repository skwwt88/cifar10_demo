import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from collections import namedtuple, defaultdict

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

batch_size = 512
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1)),
     transforms.RandomErasing(0.3, scale=(0.02, 0.1))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = InceptionModule(3, 64) 
        self.res1 = Residual(128)
        self.conv2 = ConvBn(128, 256)
        self.conv3 = ConvBn(256, 512)
        self.res3 = Residual(512)
        self.conv4 = ConvBn(512, 1024)
        self.res4 = Residual(1024)
        self.pool2 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(4)
        self.dropout = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(1024 * 1 * 1, 10, bias = False)

    def forward(self, x):
        x = self.pool2(self.conv1(x))
        x = self.res1(x)
        x = self.pool2(self.conv2(x))
        x = self.pool2(self.conv3(x))
        x = self.res3(x)

        x = self.pool2(self.conv4(x))
        x = self.res4(x)
        x = self.pool2(x)

        x = self.dropout(x)
        x = x.view(-1, 1024 * 1 * 1)
        x = self.fc1(x)
        return x

net = Net().to(device=device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-4 * batch_size, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    
    scheduler.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('Finished Training')

