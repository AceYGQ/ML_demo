"""
    Net
"""

import math
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as f
import torch.optim as optim
import time

start = time.time()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(27, 54, kernel_size=3, stride=1, padding=0)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2 = nn.Conv2d(54, 108, kernel_size=3, stride=1, padding=0)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(108, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=3, stride=1)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(1024, 512)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 64)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 4)

        self.__init_weight()

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pooling1(x)

        x = f.relu(self.conv2(x))
        x = self.pooling2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pooling3(x)

        x = x.view(-1, 1024)
        x = f.relu(self.fc1(x))
        x = self.drop1(x)
        x = f.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        x = f.softmax(x, dim=1)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)


def train(data, label, device):
    # print(device)
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    epochs = 400
    Loss = []
    target = label.squeeze().to(device)

    for epoch in range(epochs):
        net.zero_grad()
        output_data = net(data)
        loss = criterion(output_data, target)
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        Loss.append(running_loss)
        print("epoch : ", epoch)
        print("Loss : ", running_loss)

    print('Finished Training!\n\nTotal cost time: %.8f with %d training epochs' % (time.time() - start, epochs))

    # record loss
    with open("loss.txt", mode='w') as output:
        for i, l in enumerate(Loss):
            print('%d,%.8f' % (i, l), file=output)

    return net
