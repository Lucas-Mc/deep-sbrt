# Lucas McCullum
# December 27, 2020

# Re-create the original U-Net architecture in PyTorch
import torch
import torch.nn as nn
import torchvision
# For reading TIFF files / plotting
import matplotlib.pyplot as plt
# General packages
import numpy as np


TEST_IMAGE = 'images/t000.tif'


def open_image(image_path):
    image_mat = plt.imread(image_path)
    return image_mat


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, channels=(1,64,128,256,512,1024)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        filters = []
        for block in self.encoder_blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.decoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.up_convs[i](x)
            encoder_filters = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_filters], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, encoder_filters, x):
        _, _, H, W = x.shape
        encoder_filters = torchvision.transforms.CenterCrop([H, W])(encoder_filters)
        return encoder_filters


class UNet(nn.Module):
    def __init__(self,
                 encoder_channels=(1,64,128,256,512,1024),
                 decoder_channels=(1024, 512, 256, 128, 64),
                 num_class=1,
                 retain_dim=False,
                 output_size=(572,572)):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        encoder_filters = self.encoder(x)
        out = self.decoder(encoder_filters[::-1][0], encoder_filters[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, output_size)
        return out


if __name__ == "__main__":
    # Read the input image as numpy array
    input_image = open_image(TEST_IMAGE)
    # Convert the input image numpy array to PyTorch tensor
    input_image = torch.from_numpy(input_image)
    # Make the input image PyTorch tensor the correct shape
    crop_image = input_image[None, None, :572, :572]
    input_image = torch.zeros(1, 1, 572, 572)
    input_image[:, :, :crop_image.shape[2], :crop_image.shape[3]] = crop_image
    # image = torch.rand(1, 1, 572, 572)
    output_model = UNet()(input_image)
    output_image = output_model.cpu().detach().numpy()
    input_image = input_image.cpu().detach().numpy()
    plt.subplot(211)
    plt.imshow(input_image[0,0,:,:])
    plt.subplot(212)
    plt.imshow(output_image[0,0,:,:])
    plt.show()

