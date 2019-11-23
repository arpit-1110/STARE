import numpy as np 
from scipy.spatial.distance import cdist    
from utils import *
import glob
# from cv2 import imsave
from matplotlib import pyplot as plt 
class KNNSolver():
	def __init__(self,k,p,b,data):
		# k - param for KNN
		# p - param for initial number of points
		# b - param for initial number of points
		# data- numpy matrix, NxD
		# self.pointset contains indices of points considered
		# self.label_set contains corresponding labels 
		self.k = k #TODO
		self.p = p
		self.data = data 
		self.b = b
		self.point_set = []
		self.label_set = []
		self.initialize()
		print("Initialized")

	def initialize(self):
		indices = np.random.permutation(self.k)[:self.p].tolist()
		point_set = self.data[indices].tolist()
		label_set = np.random.binomial(n=1,p=0.5,size=(self.p,)).tolist()
		for x,idx in enumerate(indices):
			self.point_set.append(x)
			self.label_set.append(label_set[idx])
			dist_mat = np.linalg.norm(self.data-self.data[x,:],axis=1)
			#Appending b closest neighbours
			self.point_set+=[x for x in dist_mat.argsort(axis=0)[1:self.b+1]]
			self.label_set+=[label_set[idx] for x in range((self.b))]
		
	def step(self):
		new_point_set = [x for x in self.point_set]
		new_label_set = [x for x in self.label_set]
		#Initializing all points with seed clusters
		for idx in range(len(self.data)):
			if idx in self.point_set:
				continue
			else:
				new_point_set.append(idx)
				new_label_set.append(self.getLabel(self.data[idx]))
		self.point_set = [x for x in new_point_set]
		self.label_set = [x for x in new_label_set]
		#Main loop to label
		while True:
			print("Iterating")
			new_label_set = [x for x in self.label_set]
			for x,idx in enumerate(self.point_set):
				new_label_set[idx] = self.getLabel(x)
			if set(new_label_set) == set(self.label_set):
				break
			self.label_set = [x for x in new_label_set]
		#Majority label should be 0
		if np.array(self.label_set).sum() > len(self.label_set)//2:
			self.label_set = (1-np.array(self.label_set)).tolist()

		ret = np.zeros((self.data.shape[0],))
		ret[self.point_set] = self.label_set
		return ret    
	def getLabel(self,point,distance='euclidean'):
		#point_set is a NxD numpy array. 
		#label_set is a Nx1 array of 0/1
		#point is a 1xD numpy array
		#Function returns a label for point
		if distance == 'euclidean':
			dist_mat = np.linalg.norm(np.array(self.data[self.point_set])-point,axis=1)
			return int(np.array(self.label_set)[dist_mat.argsort(axis=0)[1:self.k+1]].sum()>(self.k)//2)
		else:
			raise NotImplementedError

def main():
	feats = []
	# adjust_gamma(path="../data/images/im0002.ppm")
	feats = extractFeature(read_img(path="../data/images/im0002.ppm",gray=False))
	feats = np.array(feats)
	feats = feats - np.mean(feats, axis = 0)
	feats /= np.std(feats, axis = 0)# 	c,h,w = feats.shape
	# feats = feats.reshape((h*w,c))
	knn = KNNSolver(k=10,p=20,b=5,data=feats)
	labels = knn.step()
	plt.imshow(labels.reshape((200,200)))
	plt.show()
main()

# feats = extractFeature(I)
# feats = np.array(feats)
# c,h,w = feats.shape
# f1 = feats[0, :, :].reshape((h*w, 1))
# f2 = feats[1, :, :].reshape((h*w, 1))
# f3 = feats[2, :, :].reshape((h*w, 1))
# feats = np.concatenate((f1, f2, f3), axis = 1)
