import torch
import torch.nn as nn

class SegModel(nn.Module):
    def __init__(self, n_channels):
        super(SegModel, self).__init__()
        self.l1 = nn.Linear(1, 128)
        self.l2 = nn.Linear(128, 512)
        self.l3 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        # nn.functional.softmax()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        # x = self.relu(self.down2(x))
        # x = self.relu(self.down3(x))
        # x = self.relu(self.up3(x))
        # x = self.relu(self.up2(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.l3(x))
        # x = self.softmax(x)
        # print(x)
        return x