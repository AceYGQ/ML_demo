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

        self.conv1 = nn.Conv2d(27, 16, kernel_size=5, stride=1, padding=0)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(3200, 512)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 64)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 1)

        self.__init_weight()

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pooling1(x)

        x = f.relu(self.conv2(x))
        x = self.pooling2(x)

        x = x.view(-1, 3200)
        x = f.relu(self.fc1(x))
        x = self.drop1(x)
        x = f.relu(self.fc2(x))
        x = self.drop2(x)
        # x = torch.sigmoid(self.fc3(x)) * 4
        x = torch.sigmoid(self.fc3(x)) * 8

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
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    epochs = 5000
    Loss = []
    running_loss = 0.0
    for epoch in range(epochs):
        print("epoch : ", epoch)
        target = label.reshape((-1, 1)).to(device)
        # print("epoch = %d, target = %d" % (epoch, target))

        net.zero_grad()
        output_data = net(data)
        loss = criterion(output_data, target)
        # print("output : ", output_data)
        # print("target : ", target)
        # print("gap : ", target - output_data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        Loss.append(running_loss)
        print("Loss : ", running_loss)

        running_loss = 0.0

    print('Finished Training!\n\nTotal cost time: %.8f with %d training epochs' % (time.time() - start, epochs))

    with open("loss.txt", mode='w') as output:
        for i, l in enumerate(Loss):
            print('%d,%.8f' % (i, l), file=output)

    return net
