import torch
import torch.nn as nn
from torchsummary import summary

def conv_layer(in_channels, out_channels, kernel_size):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
        nn.ReLU(inplace=True)
    )
    return conv 


class SurvivalNet(nn.Module):
    def __init__(self):
        super(SurvivalNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = conv_layer(1, 24, 5)
        self.down_conv_2 = conv_layer(24, 32, 3)
        self.down_conv_3 = conv_layer(32, 48, 3)
        self.down_conv_4 = conv_layer(48, 48, 3)

    def forward(self, image):
        layer_1 = self.down_conv_1(image)
        layer_2 = self.max_pool_2x2(layer_1)
        layer_3 = self.down_conv_2(layer_2)
        layer_4 = self.max_pool_2x2(layer_3)
        layer_5 = self.down_conv_3(layer_4)
        layer_6 = self.max_pool_2x2(layer_5)
        layer_7 = self.down_conv_4(layer_6)
        layer_8 = self.max_pool_2x2(layer_7)

if __name__ == '__main__':
    image = torch.rand((1, 1, 128, 128))
    model = SurvivalNet()
    summary(model, input_size=(1, 128, 128))
