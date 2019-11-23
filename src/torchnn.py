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
from models import NNModel
from data_loader import STARE
import torch.nn.functional as F
import signal
import sys

np.random.seed(42)

torch.set_default_tensor_type(torch.FloatTensor)


train_set = STARE()
dataloaders = {x: torch.utils.data.DataLoader(
	train_set, batch_size=256, shuffle=True, num_workers=0)for x in range(1)}

dataset_size = len(train_set)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=10):

	if isinstance(model, str):
		model = torch.load(model)

	def save_model(*args):
		torch.save(model, './Models/model')
		sys.exit(0)

	signal.signal(signal.SIGINT, save_model)

	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch ' + str(epoch+1) + ' running')
		if epoch > 15:
			optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.1)
		# if epoch > 30:
		# 	optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.01)
		# if epoch > 40:
		# 	optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.01)
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
	
model = NNModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 5])).float())
# criterion = nn.CrossEntropyLoss()
model_optim = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.1)
model = train_model(model, criterion, model_optim,
                    # exp_lr_scheduler,
                    num_epochs=20)

torch.save(model, './Models/model')

