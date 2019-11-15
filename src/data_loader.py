import torch
import torch.utils.data
import numpy as np
from PIL import Image
import glob
from utils import extractFeature
import cv2

X_size = (605, 700)
y_size = (605, 700)

class STARE(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		# self.X_train = glob.glob('../data/images/*.ppm')
		# self.y_train = glob.glob('../data/labels/*.ppm')
		self.X_train = cv2.imread("../data/images/im0001.ppm")
		self.X_train = cv2.cvtColor(self.X_train, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
		# self.X_train = self.X_train/255
		# cv2.imshow(self)
		self.X_train = self.X_train.reshape(605*700)
		# self.X_train = extractFeature(self.X_train).reshape(605*700, 3)
		# cv2.imwrite('temp.png', self.X_train[:, 2].reshape(605, 700))
		self.y_train = cv2.imread("../data/labels/im0001.ah.ppm", cv2.IMREAD_GRAYSCALE).reshape(605, 700)
		# self.y_train = cv2.cvtColor(self.X_train, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
		self.y_train = self.y_train//255
		self.y_train = np.eye(2)[self.y_train].reshape(605*700, 2)

		idx = np.random.permutation(605*700)
		self.X_train = self.X_train[idx]
		self.y_train = self.y_train[idx]

	def __len__(self):
		# print(self.X_train)
		return 605*700

	def __getitem__(self, idx):
		X = self.X_train[idx].reshape(1)

		y = self.y_train[idx].reshape(2)

		# print(X.shape, y.shape)
		# print(np.max(X), np.max(y))
		# exit()
		# X = X/255.0
		# y = y/255.0
		return torch.from_numpy(X).float(), torch.from_numpy(y).float()