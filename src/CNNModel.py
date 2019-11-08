import torch
import torch.nn as nn

class SegModel(nn.Module):
    def __init__(self, n_channels):
        super(SegModel, self).__init__()
        self.down1 = nn.Conv2d(n_channels, 16, (7, 7))
        self.down2 = nn.Conv2d(8, 8, (5, 5))
        self.down3 = nn.Conv2d(8, 16, (3, 3))
        self.up3 = nn.ConvTranspose2d(16, 8, (3, 3))
        self.up2 = nn.ConvTranspose2d(8, 8, (5, 5))
        self.up1 = nn.ConvTranspose2d(16, n_channels, (7, 7))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.down1(x))
        # x = self.relu(self.down2(x))
        # x = self.relu(self.down3(x))
        # x = self.relu(self.up3(x))
        # x = self.relu(self.up2(x))
        x = self.sigmoid(self.up1(x))
        return x