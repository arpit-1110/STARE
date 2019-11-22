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
		timgs = glob.glob('../data/images/*.ppm')
		tlbls = glob.glob('../data/labels/*.ppm')
		n_img = len(timgs)
		idx = np.random.permutation(n_img)
		imgs = [None]*n_img
		lbls = [None]*n_img
		for i in range(n_img):
			imgs[i] = timgs[idx[i]]
			lbls[i] = tlbls[idx[i]]
		def init_helper(img, lbl):
			image = cv2.imread(img)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
			image = extractFeature(image, 85.1, 48.9).reshape(size, 3)
			label = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE).reshape(605, 700)
			label = label//255
			label = np.eye(2)[label].reshape(size, 2)
			return image, label

		self.X_train = np.zeros((size*len(imgs)//2, 3))
		self.y_train = np.zeros((size*len(imgs)//2, 2))
		for i in range(len(imgs)//2):
			self.X_train[size*i:size*(i+1), :], self.y_train[size*i:size*(i+1), :] = init_helper(imgs[i], lbls[i])
		idx = np.random.permutation(size*len(imgs)//2)
		self.X_train = self.X_train[idx]
		self.y_train = self.y_train[idx]


	def __len__(self):
		return size

	def __getitem__(self, idx):
		X = self.X_train[idx].reshape(3)

		y = self.y_train[idx].reshape(2)
		return torch.from_numpy(X).float(), torch.from_numpy(y).float()


class STARECNN(torch.utils.data.Dataset):
	def __init__(self, transform=None):
		self.X_train = glob.glob('../data/images/*.ppm')
		self.y_train = glob.glob('../data/labels/*.ppm')


	def __len__(self):
		return len(self.X_train)

	def __getitem__(self, idx):
		X = self.X_train[idx]
		X = cv2.imread(X)
		X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
		X = extractFeature(X, 85.1, 48.9)[:, 0].reshape((1, 605, 700))

		y = cv2.imread(self.y_train[idx], cv2.IMREAD_GRAYSCALE).reshape((1, 605, 700))
		y = y//255
		return torch.from_numpy(X).float(), torch.from_numpy(y).float()