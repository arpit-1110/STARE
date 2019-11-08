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


def dice_coeff(inputs, target):
	eps = 1e-7
	coeff = 0
	# print(inputs.shape)
	for i in range(inputs.shape[0]):
		iflat = inputs[i,:,:,:].view(-1)
		tflat = target[i,:,:,:].view(-1)
		# print(max(iflat))
		# print(max(tflat))
		# exit()
		# print(torch.max(iflat), torch.min(iflat))
		# print(torch.max(tflat), torch.min(tflat))
		intersection = torch.dot(iflat, tflat)
		# print('intersection	', 2*intersection)
		# print(iflat.sum() + tflat.sum())
		coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
	# print((2. * intersection) / (iflat.sum() + tflat.sum()))
	return coeff/inputs.shape[0]

def dice_loss(inputs, target):
	return 1 - dice_coeff(inputs, target)

train_set = STARE('train')
dataloaders = {x: torch.utils.data.DataLoader(
	train_set, batch_size=2, shuffle=True, num_workers=0)for x in range(2)}

dataset_size = len(train_set)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=10):
	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch ' + str(epoch) + ' running')
		if epoch > 30:
			optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.1)
		if epoch > 50:
			optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.1)
		if epoch > 75:
			optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.1)
		model.train()
		running_loss = 0.0
		val_dice = 0
		count = 0

		for i, Data in enumerate(train_set):
				count += 1
				inputs, masks = Data
				# print('SHAPE', inputs.shape)
				inputs = inputs.to(device)
				masks = masks.to(device)
				inputs, masks = Variable(inputs), Variable(masks)
				# print(masks.shape)
				optimizer.zero_grad()
				with torch.set_grad_enabled(True):
					pred_mask = model(inputs)
					# pred_mask = (pred_mask-torch.min(pred_mask))/(torch.max(pred_mask) - torch.min(pred_mask))
					# print(pred_mask.shape)
					if not i % 4:
						t = ToPILImage()
						a = t(pred_mask[0].cpu().detach())
						# a.save('./Results/result_' + str(i) +
						# 	   'epoch' + str(epoch) + '.png')
						a.save('./Results/result_' + str(i) + '.png')
					# print(pred_mask.shape)
					if criterion is not None:
						pred_temp_mask = pred_mask.view(-1)
						temp_masks = masks.view(-1)
						# print(pred_mask)

						loss = criterion(pred_temp_mask, temp_masks)
					else:
						# print(masks.shape, pred_mask.shape)
						# print(pred_mask)

						# print(torch.max(pred_mask), torch.min(pred_mask))
						loss = dice_loss(masks, pred_mask)
						# print("loss in batch ",loss)
					val_dice += dice_coeff(masks, pred_mask)
					# if phase == 0:
					loss.backward()
					optimizer.step()
				running_loss += loss.item()
		epoch_loss = running_loss / dataset_size
		print('Epoch finished ! Loss: {}'.format(epoch_loss))
		# epoch_acc = running_corrects.double() / dataset_sizes[phase]
		# val_dice = 
		# val_dice = eval_net(model, val_set, torch.cuda.is_available())
		print('Training Dice Coeff: {}'.format(val_dice/count))

		print('End of epoch')
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	# print('Best val Acc: {:4f}'.format(best_acc))
	# model.load_state_dict(best_model_wts)
	return model
	
model = SegModel(1)
model = model.to(device)
criterion = nn.MSELoss()
model_optim = optim.SGD(model.parameters(), lr=5e-1, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.1)
model = train_model(model, None, model_optim,
                    # exp_lr_scheduler,
                    num_epochs=100)
torch.save(model, './Models/model')

