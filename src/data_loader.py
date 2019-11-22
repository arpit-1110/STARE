import torch
import torch.utils.data
import numpy as np
from PIL import Image
import glob
from utils import extractFeature
import cv2

X_size = (605, 700)
y_size = (605, 700)
size = 605*700

class STARE(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		imgs = glob.glob('../data/images/*.ppm')
		lbls = glob.glob('../data/labels/*.ppm')
		def init_helper(img, lbl):
			image = cv2.imread(img)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
			# self.X_train = self.X_train/255
			# cv2.imshow(self)
			image = extractFeature(image, 85.1, 48.9).reshape(size, 3)
			# cv2.imwrite('temp.png', image[:, 2].reshape(605, 700))
			label = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE).reshape(605, 700)
			# self.y_train = cv2.cvtColor(self.X_train, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
			label = label//255
			label = np.eye(2)[label].reshape(size, 2)
			return image, label

		self.X_train = np.zeros((size*len(imgs)//2, 3))
		self.y_train = np.zeros((size*len(imgs)//2, 2))
		for i in range(len(imgs)//2):
			self.X_train[size*i:size*(i+1), :], self.y_train[size*i:size*(i+1), :] = init_helper(imgs[i], lbls[i])
		idx = np.random.permutation(size)
		self.X_train = self.X_train[idx]
		self.y_train = self.y_train[idx]


	def __len__(self):
		# print(self.X_train)
		return size

	def __getitem__(self, idx):
		X = self.X_train[idx].reshape(3)

		y = self.y_train[idx].reshape(2)

		# print(X.shape, y.shape)
		# print(np.max(X), np.max(y))
		# exit()
		# X = X/255.0
		# y = y/255.0
		return torch.from_numpy(X).float(), torch.from_numpy(y).float()