import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,ReLU,BatchNormalization,Conv2DTranspose,MaxPool2D,Dense,Flatten,Layer
from utils import get_dataset
import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

class MaxPool2D(Layer):

	def __init__(
			self,
			ksize=(2, 2),
			strides=(2, 2),
			padding='same',
			**kwargs):
		super(MaxPool2D, self).__init__(autocast=False)
		self.padding = padding
		self.pool_size = ksize
		self.strides = strides

	def call(self, inputs, **kwargs):
		padding = self.padding
		pool_size = self.pool_size
		strides = self.strides
		ksize = [1, pool_size[0], pool_size[1], 1]
		padding = padding.upper()
		strides = [1, strides[0], strides[1], 1]
		output, argmax = tf.nn.max_pool_with_argmax(
				inputs,
				ksize=ksize,
				strides=strides,
				padding=padding)
		argmax = tf.cast(argmax, tf.float64)
		return [output, argmax]

	def compute_output_shape(self, input_shape):
		ratio = (1, 2, 2, 1)
		output_shape = [
				dim//ratio[idx]
				if dim is not None else None
				for idx, dim in enumerate(input_shape)]
		output_shape = tuple(output_shape)
		return [output_shape, output_shape]

	def compute_mask(self, inputs, mask=None):
		return 2 * [None]


class MaxUnpool2D(Layer):
	def __init__(self, ksize=(2, 2), **kwargs):
		super(MaxUnpool2D, self).__init__(autocast=False,**kwargs)
		self.size = ksize

	def call(self, inputs, output_shape=None):
		updates, mask = inputs[0], inputs[1]
		mask = tf.cast(mask, 'int32')
		input_shape = tf.shape(updates, out_type='int32')
		#  calculation new shape
		if output_shape is None:
			output_shape = (
					input_shape[0],
					input_shape[1]*self.size[0],
					input_shape[2]*self.size[1],
					input_shape[3])
		self.output_shape1 = output_shape

		# calculation indices for batch, height, width and feature maps
		one_like_mask = tf.ones_like(mask, dtype='int32')
		batch_shape = tf.concat(
				[[input_shape[0]], [1], [1], [1]],
				axis=0)
		batch_range = tf.reshape(
				tf.range(output_shape[0], dtype='int32'),
				shape=batch_shape)
		# print("SHAPE______",output_shape[3])
		b = one_like_mask * batch_range
		y = mask // (output_shape[2] * output_shape[3])
		x = (mask // output_shape[3]) % output_shape[2]
		feature_range = tf.range(output_shape[3], dtype='int32')
		f = one_like_mask * feature_range

		# transpose indices & reshape update values to one dimension
		updates_size = tf.size(updates)
		indices = tf.transpose(tf.reshape(
			tf.stack([b, y, x, f]),
			[4, updates_size]))
		values = tf.reshape(updates, [updates_size])
		ret = tf.scatter_nd(indices, values, output_shape)
		return ret

	def compute_output_shape(self, input_shape):
		mask_shape = input_shape[1]
		return (
				mask_shape[0],
				mask_shape[1]*self.size[0],
				mask_shape[2]*self.size[1],
				mask_shape[3]
				)
class segnet(tf.keras.Sequential):
	def __init__(self):
		super(segnet,self).__init__()
	def conv_layer(self, channel):
		conv_block = tf.keras.Sequential(
			[Conv2D(filters=channel, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),
			BatchNormalization(axis=-1),
			ReLU()]
		)
		return conv_block

class encoder(segnet):
	def __init__(self,channels=3):
		super(encoder,self).__init__()
		filter = [64, 128, 256, 512, 512]
		self.conv_block_enc = []
		self.conv_block_enc.append(Sequential([self.conv_layer(filter[0]),self.conv_layer(filter[0])]))
		for i in range(4):  #TODO Refactor for better model making
			if i == 0:
				self.conv_block_enc.append(Sequential([self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1])]))
			else:
				self.conv_block_enc.append(Sequential([self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1])]))
		self.down_sampling = MaxPool2D(ksize=(2,2),padding='same')
	def call(self,x):
		x1 = x
		indices = []
		sizes = []
		for i in range(5):
			x1 = self.conv_block_enc[i](x1)
			sizes.append(x1.shape)
			x1,index = self.down_sampling.call(x1)
			indices.append(index)
			#print("Encode",x1.shape,indices[-1].shape)
		encout = x1
		indices = indices
		return x1,indices,sizes

class decoder(segnet):
	def __init__(self,channels=3):
		super(decoder,self).__init__()
		filter = [64, 128, 256, 512, 512]
		self.conv_block_dec = []
#       self.conv_block_dec = Sequential()
		for i in range(1,4):
			self.conv_block_dec.append(Sequential([self.conv_layer(filter[-i]),
												  self.conv_layer(filter[-i]),
												  self.conv_layer(filter[-(i+1)])]))

		self.conv_block_dec.append(Sequential([self.conv_layer(filter[1]),
												  self.conv_layer(filter[0])]))
		self.conv_block_dec.append(Sequential([self.conv_layer(filter[0]),
												  tf.keras.Sequential(
			[BatchNormalization(axis=-1),
			Conv2D(filters=1, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid"),]
		)]))    #Getting best results when sigmoid, batch_norm, relu
		
		self.up_sampling = MaxUnpool2D(ksize=(2,2))
	def forward(self,X,indices,sizes):
		indices = indices[::-1]
		sizes = sizes[::-1]
		for idx,layer in enumerate(self.conv_block_dec):
			#print(X.shape,indices[idx].shape)
			# print(idx,X.shape,self.max_indices[idx].shape)
			X = self.up_sampling.call([X,indices[idx]],sizes[idx])
			#print(idx,X.shape,indices[idx].shape)
			X = layer(X) 
		return X

class Model(tf.keras.Sequential):
	def __init__(self):
#       encoder is a model of type encoder defined above,
#       decoders is a list of decoders 
		super(Model,self).__init__()
		self.encoder = encoder()
		self.decoder = decoder()
		self.TrainableVarsSet = False
		self.optimizer = tf.keras.optimizers.Adam()
	def setTrainableVars(self):
		self.TrainableVars = self.encoder.trainable_variables+self.decoder.trainable_variables
	def call(self,X):
		X,indices,sizes = self.encoder.call(X)
		return self.decoder.forward(X,indices,sizes)
		

	def loss_classification(self,y,labels,beta=10):
		return (-1*tf.reduce_mean(beta*labels*(tf.math.log(y+1e-5)) + (1-labels)*(tf.math.log(1-y+1e-5))))
	
	
	@tf.function
	def train_on_batch(self,X,labels):
		if self.TrainableVarsSet == False:
			self.setTrainableVars()
		with tf.GradientTape() as tape:
			y = self.call(X)
			loss = self.loss_classification(y,tf.cast(labels,tf.float32))
		grads = tape.gradient(loss,self.TrainableVars)
		
		#tf.clip_by_value(self.discTrainableVariables,-0.01,0.01)
		grads_and_vars = zip(grads, self.TrainableVars)
		self.optimizer.apply_gradients(grads_and_vars)
		#print(grads_and_vars_gen)
		return loss,y

	def save(self):
		pickle.dump(self.encoder.trainable_variables,open("EncWeights","wb"))
		pickle.dump(self.decoder.trainable_variables,open("DecWeights","wb"))

	def load_model(self):
		enc_train_vars = pickle.load(open("EncWeights","rb"))
		dec_train_vars = pickle.load(open("DecWeights","rb"))
		for l in self.encoder.layers:
		    # weights = l.get_weights()
		    weights = enc_train_vars
		    l.set_weights(weights)
		for l in self.decoder.layers:
		    # weights = l.get_weights()
		    weights = dec_train_vars
		    l.set_weights(weights)
		self.TrainableVarsSet = False

def train(model,X,labels,validation,epochs,batchsize):
	for i in range(epochs):
		idx = np.random.permutation(len(X))
		X = X[idx]
		labels = labels[idx]
		loss = []
		for j in tqdm(range(len(X)//batchsize)):
			l,y = (model.train_on_batch(X[j*batchsize:(j+1)*batchsize],np.expand_dims(labels[j*batchsize:(j+1)*batchsize],-1)))
			loss.append(l)
			# np.savetxt("TeOut-{}-{}".format(i,j),y.numpy()[0,:,:,0])
		print("Epoch-{},Loss-{}".format(i,np.array(loss).mean()))
		model.save()
		lab_val = model.call(validation)
		plt.imshow(lab_val[0,:,:,0])
		plt.savefig("Result/Epoch-{}.png".format(i))
		plt.close()

def main():
	X,labels = get_dataset('images/train','labels/train')
	X_val,lab_val = get_dataset('images/val','labels/val')
	X_val = X_val[:1]
	plt.imshow(lab_val[0,:,:])
	plt.savefig("Actual.png")
	plt.close()

	model = Model()
	model.call(X_val)
	train(model,X,labels,X_val,100,3)

if __name__ == '__main__':
	main()
