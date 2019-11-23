import numpy as np
import torch
import torch.nn as nn
from torch import optim
import time
from torch.autograd import Variable
from nn import NNModel
from data_loader import STARE
import torch.nn.functional as F
import signal
import sys

np.random.seed(42)

torch.set_default_tensor_type(torch.FloatTensor)

batch_size = 128

train_set = STARE()
dataloaders = {x: torch.utils.data.DataLoader(
	train_set, batch_size=batch_size, shuffle=True, num_workers=0)for x in range(1)}

dataset_size = len(train_set)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=10):

	def save_model(*args):
		torch.save(model, './Models/model')
		sys.exit(0)

	signal.signal(signal.SIGINT, save_model)

	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch ' + str(epoch+1) + ' running')
		if epoch > 20:
			optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.1)
		if epoch > 40:
			optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.05)
		if epoch > 60:
			optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.01)
		if epoch > 20:
			optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.01)
		model.train()
		running_loss = 0.0
		count = 0
		for i, Data in enumerate(dataloaders[0]):
			count += 1
			inputs, labels = Data
			inputs = inputs.to(device)
			labels = labels.to(device)
			inputs, labels = Variable(inputs), Variable(labels)
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
		epoch_loss = running_loss / count
		print('Epoch finished ! Loss: {}'.format(epoch_loss))

		print('End of epoch')
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	
	return model
	
model = NNModel()
if not isinstance(model, str):
	model = model.to(device)
else:
	model = torch.load(model)

criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 5])).float())
# optim.Adam()
model_optim = optim.SGD(model.parameters(), lr=4e-2, momentum=0.9)
model = train_model(model, criterion, model_optim,
                    # exp_lr_scheduler,
                    num_epochs=100)

torch.save(model, './Models/model')

