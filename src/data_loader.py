import torch
import torch.utils.data
import numpy as np
from scipy.misc import imshow, imsave
import glob
from utils import extractFeature
import cv2

X_size = (605, 700)
y_size = (605, 700)
size = 605*700

class STARE(torch.utils.data.Dataset):
	def __init__(self):
		imgs = glob.glob('../data/images/*.ppm')
		lbls = glob.glob('../data/labels/*.ppm')

		def init_helper(img, lbl):
			image = cv2.imread(img)
			image = image.reshape(605, 700, 3)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, :, 1].reshape(605, 700)
			image = extractFeature(image, 85.1, 48.9).reshape(size, 3)
			label = cv2.imread(lbl, cv2.IMREAD_GRAYSCALE).reshape(605, 700)
			label = label//255
			label = np.eye(2)[label].reshape(size, 2)
			return image, label

		self.X_train = np.zeros((size*len(imgs), 3))
		self.y_train = np.zeros((size*len(imgs), 2))
		for i in range(len(imgs)):
			self.X_train[size*i:size*(i+1), :], self.y_train[size*i:size*(i+1), :] = init_helper(imgs[i], lbls[i])
		idx = np.random.permutation(size*len(imgs))
		self.X_train = self.X_train[idx]
		self.y_train = self.y_train[idx]


	def __len__(self):
		return size

	def __getitem__(self, idx):
		X = self.X_train[idx].reshape(3)

		y = self.y_train[idx].reshape(2)
		return torch.from_numpy(X).float(), torch.from_numpy(y).float()
