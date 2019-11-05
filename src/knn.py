import numpy as np 
from scipy.spatial.distance import cdist	
from utils import *
import glob

class KNNSolver():
	def __init__(self,k,p,b,data):
		# k - param for KNN
		# p - param for initial number of points
		# b - param for initial number of points
		# data- numpy matrix, NxD
		# self.pointset contains indices of points considered
		# self.label_set contains corresponding labels 
		self.k = k
		self.p = p
		self.data = data 
		self.b = b
		self.point_set = []
		self.label_set = []
		# self.distances = cdist(self.data,self.data).argsort(axis=0)[1:k]
		# print(self.distances.shape)
		# print("Got dist")
		self.initialize()
		print("Initialized")

	def initialize(self):
		indices = np.random.permutation(self.k)[:self.p].tolist()
		point_set = self.data[indices].tolist()
		label_set = np.random.binomial(n=1,p=0.5,size=(self.p,)).tolist()
		for x,idx in enumerate(indices):
			self.point_set.append(x)
			self.label_set.append(label_set[idx])
			dist_mat = np.linalg.norm(np.array(point_set)-self.data[x,:],axis=1)
			#Appending b closest neighbours
			self.point_set+=[x for x in dist_mat.argsort(axis=0)[1:self.b+1]]
			self.label_set+=[label_set[idx] for x in range((self.b))]
		
	def step(self):
		new_point_set = [x for x in self.point_set]
		new_label_set = [x for x in self.label_set]
		for idx in range(len(self.data)):
			if idx in self.point_set:
				continue
			else:
				new_point_set.append(idx)
				new_label_set.append(self.getLabel(self.data[idx]))
		self.point_set = [x for x in new_point_set]
		self.label_set = [x for x in new_label_set]
		while True:
			print("Iterating")
			new_label_set = [x for x in self.label_set]
			for x,idx in enumerate(self.point_set):
				print(idx)
				new_label_set[idx] = self.getLabel(x)
			if set(new_label_set) == set(self.label_set):
				break
			self.label_set = [x for x in new_label_set]
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
	# for img in glob.glob("../data/images/*.ppm"):
	# 	feats.append(extractFeature(read_img(img,gray=True)))
	# 	print(img)
	feats = extractFeature(read_img("../data/images/im0002.ppm",gray=True))
	feats = np.array(feats)
	# print(feats.shape)
	c,h,w = feats.shape
	feats = feats.reshape((h*w,c))
	knn = KNNSolver(k=10,p=50,b=10,data=feats)
	labels = knn.step()
main()