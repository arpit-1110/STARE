import torch
import torch.utils.data
import numpy as np
from PIL import Image
import glob

X_size = (605, 700)
y_size = (605, 700)

class STARE(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.X_train = glob.glob('../data/images/*.ppm')
        self.y_train = glob.glob('../data/labels/*.ppm')

    def __len__(self):
        # print(self.X_train)
        return len(self.X_train)

    def __getitem__(self, idx):
        X = np.array(Image.open(self.X_train[idx]).convert(
            'L').resize(X_size)).reshape(1, 1, 605, 700)

        y = np.array(Image.open(self.y_train[idx]).convert(
            'L').resize(y_size)).reshape(1, 1, 605, 700)

        # print(X.shape, y.shape)
        # print(np.max(X), np.max(y))
        # exit()
        X = X/255.0
        y = y/255.0
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()