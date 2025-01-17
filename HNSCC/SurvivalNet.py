import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from get_data import SurvivalDataset, Rescale, ToTensor


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
        nn.ReLU(inplace=False)#True)
    )
    return fc

def dropout_layer():
    """
    p = 0.5
    """
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

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        w = m.weight.data
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
        # TODO: expand this out for N images, possibly like:
        #   for image in images:
        #       layer_1 = self.down_conv_1(image)
        #       ... = ...
        #   layer_11 = self.multiple_fusion(*images)
        #   return layer_11
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
        return layer_11


def load_split_train_test(train_split, batch_size, num_workers):
    # Generate the random test and train datasets
    transform = torchvision.transforms.Compose([Rescale(128), ToTensor()])
    train_data = SurvivalDataset('train', train_split, transform=transform)
    test_data = SurvivalDataset('test', train_split, transform=transform)
    # Turn the data into `DataLoader` objects with desired batch size and
    # number of workers
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=num_workers)
    return train_loader, test_loader


def training_loop(n_epochs, batch_size, learning_rate, optimizer, model,
                  loss_fn, t_u_train, t_c_train, t_u_test, t_c_test):
    total_train_losses = []
    total_test_losses = []
    total_train_accuracy = []
    total_test_accuracy = []

    for epoch in range(1, n_epochs+1):
        train_loss = 0
        train_accuracy = 0

        for i in range(len(t_u_train)):
            t_p_train = model.forward(torch.tensor(np.expand_dims(t_u_train[i],axis=[0,1])).to(device))
            loss_train = loss_fn(t_p_train, torch.tensor([t_c_train[i]]).to(device))
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item()
            pred_val = t_p_train.tolist()[0].index(max(t_p_train.tolist()[0]))
            actual_val = t_c_train[i]
            if pred_val == actual_val:
                train_accuracy += 1

            # Change the batch size here
            if (i == len(t_u_train)-1) and ((epoch == 1) or (epoch%1 == 0)):
                total_train_losses.append(train_loss/len(t_u_train))
                total_train_accuracy.append(train_accuracy/len(t_u_train))
                print(f'Epoch {epoch}, Training loss {train_loss/len(t_u_train):.4f}, Training accuracy {train_accuracy/len(t_u_train):4f}')
                test_loss = 0
                model.eval()

                with torch.no_grad():
                    test_loss = 0
                    test_accuracy = 0

                    for i in range(len(t_u_test)):
                        t_p_test = model.forward(torch.tensor(np.expand_dims(t_u_test[i],axis=[0,1])).to(device))
                        loss_test = loss_fn(t_p_test, torch.tensor([t_c_test[i]]).to(device))
                        test_loss += loss_test.item()
                        pred_val = t_p_test.tolist()[0].index(max(t_p_test.tolist()[0]))
                        actual_val = t_c_test[i]
                        if pred_val == actual_val:
                            test_accuracy += 1

                    total_test_losses.append(test_loss/len(t_u_test))
                    total_test_accuracy.append(test_accuracy/len(t_u_test))
                    print(f'Epoch {epoch}, Test loss {test_loss/len(t_u_test):.4f}, Test Accuracy {test_accuracy/len(t_u_test):.4f}')

    plt.plot(total_train_losses, label='Train Loss')
    plt.plot(total_test_losses, label='Test Loss')
    plt.plot(total_train_accuracy, label='Train Accuracy')
    plt.plot(total_test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.title(f'Training Data: Learning Rate = {learning_rate}, Batch Size = {batch_size}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_split = 0.75
    learning_rate = 1e-6
    epochs = 50
    batch_size = 1
    num_workers = 2
    image = torch.rand((1, 1, 128, 128))
    model = SurvivalNet()
    model.apply(weights_init)
    model.to(torch.double)
    # summary(model, input_size=(1, 128, 128))
    torch.autograd.set_detect_anomaly(True)

    train_loader, test_loader = load_split_train_test(train_split, batch_size, num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_u_train = train_loader.dataset.all_sagittal
    t_c_train = train_loader.dataset.outcome
    t_u_test = test_loader.dataset.all_sagittal
    t_c_test = test_loader.dataset.outcome
    training_loop(epochs, batch_size, learning_rate, optimizer, model, criterion,
                  t_u_train, t_c_train, t_u_test, t_c_test)

    # epochs = 10
    # steps = 0
    # running_loss = 0
    # train_losses, test_losses = [], []

    # for epoch in range(1,epochs+1):
    #     for p,a,c,s,o in train_loader:
    #         # SurvivalDataset().plot_image(images=s,batch_size=batch_size)
    #         steps += 1
    #         inputs = s
    #         labels = o
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         log_ps = model.forward(inputs)
    #         # print(f'a,s,c: {criterion(model.forward(a),labels)}',\
    #         #              f'{criterion(model.forward(s),labels)}',\
    #         #              f'{criterion(model.forward(c),labels)}')
    #         loss_train = criterion(log_ps, labels)
    #         loss_train.backward()
    #         optimizer.step()
    #         running_loss += loss_train.item()

    #         if steps % batch_size == 0:
    #             print(f'Epoch {epoch}, Training loss {loss_train.item():.4f}')
    #             # test_loss = 0
    #             # accuracy = 0
    #             # model.eval()
    #             # with torch.no_grad():
    #             #     for p,a,c,s,o in test_loader:
    #             #         inputs = s
    #             #         labels = o
    #             #         inputs, labels = inputs.to(device), labels.to(device)
    #             #         log_ps = model.forward(inputs)
    #             #         batch_loss = criterion(log_ps, labels)
    #             #         test_loss += batch_loss.item()
    #             #         ps = torch.exp(log_ps)
    #             #         top_p, top_class = ps.topk(1, dim=1)
    #             #         equals = top_class == labels.view(*top_class.shape)
    #             #         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #             # train_losses.append(running_loss/len(train_loader))
    #             # test_losses.append(test_loss/len(test_loader))                    
    #             # print(f"Epoch {epoch}/{epochs}.. "
    #             #       f"Train loss: {running_loss/batch_size:.3f}.. "
    #             #       f"Test loss: {test_loss/len(test_loader):.3f}.. "
    #             #       f"Test accuracy: {accuracy/len(test_loader):.3f}")
    #             # running_loss = 0
    #             # model.train()
    # torch.save(model, 'three_view_model.pth')
