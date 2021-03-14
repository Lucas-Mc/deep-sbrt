import torch
import torch.nn as nn
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
        layer_10 = self.single_fusion(layer_9)


if __name__ == '__main__':
    image = torch.rand((1, 1, 128, 128))
    model = SurvivalNet()
    summary(model, input_size=(1, 128, 128))
