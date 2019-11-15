import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image
# import torchvision
from torchvision.transforms import ToPILImage
import time
# import copy
from torch.autograd import Variable
from CNNModel import SegModel
from data_loader import STARE
import torch.nn.functional as F

np.random.seed(42)

torch.set_default_tensor_type(torch.FloatTensor)

def dice_coeff(inputs, target):
	eps = 1e-7
	coeff = 0
	# print(inputs.shape)
	for i in range(inputs.shape[0]):
		iflat = inputs[i,:,:,:].view(-1)
		tflat = target[i,:,:,:].view(-1)
		intersection = torch.dot(iflat, tflat)
		coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
	return coeff/inputs.shape[0]

def dice_loss(inputs, target):
	return 1 - dice_coeff(inputs, target)

train_set = STARE('train')
dataloaders = {x: torch.utils.data.DataLoader(
	train_set, batch_size=128, shuffle=True, num_workers=0)for x in range(1)}

dataset_size = len(train_set)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=10):
	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch ' + str(epoch) + ' running')
		if epoch > 30:
			optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.1)
		if epoch > 50:
			optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.1)
		if epoch > 75:
			optimizer = optim.SGD(model.parameters(), lr=5e-5, momentum=0.1)
		model.train()
		running_loss = 0.0
		val_dice = 0
		count = 0
		for i, Data in enumerate(dataloaders[0]):
			# print(Data)
			count += 1
			inputs, labels = Data
			# print('SHAPE', inputs.shape)
			inputs = inputs.to(device)
			labels = labels.to(device)
			inputs, labels = Variable(inputs), Variable(labels)
			# print(labels.shape)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				pred_label = model(inputs)
				# print(pred_label[0][0])
				if criterion is not None:
					pred_temp_label = pred_label
					temp_labels = torch.max(labels.long(), 1)[1]
					loss = criterion(pred_temp_label, temp_labels)
				else:
					raise ValueError()
				loss.backward()
				optimizer.step()
			running_loss += loss.item()
		epoch_loss = running_loss / dataset_size
		print('Epoch finished ! Loss: {}'.format(epoch_loss))

		print('End of epoch')
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	return model
	
model = SegModel(1)
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 10])).float())
# criterion = nn.CrossEntropyLoss()
model_optim = optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.1)
model = train_model(model, criterion, model_optim,
                    # exp_lr_scheduler,
                    num_epochs=10)
torch.save(model, './Models/model')

