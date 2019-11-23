import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.l1 = nn.Linear(3, 16)
        self.l2 = nn.Linear(16, 64)
        self.l3 = nn.Linear(64, 128)
        # self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        # nn.functional.softmax()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.softmax(self.l4(x))
        return x

class CNNModel(nn.Module):
    def __init__(self, n_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 8, 5, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1)
        self.tconv2 = nn.ConvTranspose2d(16, 8, 3, stride=1)
        self.tconv1 = nn.ConvTranspose2d(8, n_channels, 5, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.tconv2(x))
        x = self.sigmoid(self.tconv1(x))
        return x
        