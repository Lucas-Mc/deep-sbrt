import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


def conv_layer(in_channels, out_channels, kernel_size):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
        nn.ReLU(inplace=True)
    )
    return conv

def fc_layer(in_channels, out_channels):
    fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_channels, out_channels),
        nn.ReLU(inplace=True)
    )
    return fc

def dropout_layer():
    return nn.Dropout(p=0.5, inplace=True)

def single_fusion():
    fusion = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16, 2),
        nn.Softmax(dim=1)
    )
    return fusion

def committee_fusion():
    fusion = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16, 2),
        nn.Softmax(dim=1)
    )
    return fusion

def late_fusion(in_channels):
    fusion = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16*in_channels, 2),
        nn.Softmax(dim=1)
    )
    return fusion

def mixed_fusion():
    fusion = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16*3, 2),
        nn.Softmax(dim=1)
    )
    return fusion


class SurvivalNet(nn.Module):
    def __init__(self):
        super(SurvivalNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = conv_layer(1, 24, 5)
        self.down_conv_2 = conv_layer(24, 32, 3)
        self.down_conv_3 = conv_layer(32, 48, 3)
        self.down_conv_4 = conv_layer(48, 48, 3)
        self.fc_1 = fc_layer(48*6*6, 16)
        self.dropout = dropout_layer()
        self.single_fusion = single_fusion()

    def forward(self, image):
        layer_1 = self.down_conv_1(image)
        layer_2 = self.max_pool_2x2(layer_1)
        layer_3 = self.down_conv_2(layer_2)
        layer_4 = self.max_pool_2x2(layer_3)
        layer_5 = self.down_conv_3(layer_4)
        layer_6 = self.max_pool_2x2(layer_5)
        layer_7 = self.down_conv_4(layer_6)
        layer_8 = self.max_pool_2x2(layer_7)
        layer_9 = self.fc_1(layer_8)
        layer_10 = self.dropout(layer_9)
        layer_11 = self.single_fusion(layer_10)


if __name__ == '__main__':
    image = torch.rand((1, 1, 128, 128))
    model = SurvivalNet()
    # summary(model, input_size=(1, 128, 128))
    get_data()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters())
    # nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

    # trainloader = []
    # for epoch in range(2):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 128 == 127:    # print every 128 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 128))
    #             running_loss = 0.0

    # print('Finished Training')