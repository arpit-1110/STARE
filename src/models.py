import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.l1 = nn.Linear(3, 64)
        self.l2 = nn.Linear(64, 128)
        # self.l3 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        # nn.functional.softmax()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))
        x = self.softmax(self.l3(x))
        return x