import numpy as np 

class KNNSolver():
	def __init__(self,k,p,b,data):
		#k - param for KNN
		#p - param for initial number of points
		#b - param for initial number of points
		#data- numpy matrix, NxD
		# self.pointset contains indices of points considered
		# self.label_set contains corresponding labels 
		self.k = k
		self.p = p
		self.data = data 
		self.b = b
		self.point_set = []
		self.label_set = []
		self.initialize()

	def initiailize(self):
		pass
		
	def step(self):
		new_point_set = [x for x in self.point_set]
		new_label_set = [x for x in self.label_set]
		for idx in range(len(self.data)):
			if idx in self.point_set:
				continue
			else:
				new_point_set.append(idx)
				new_label_set.append(getLabel(self.data[idx]))
		self.point_set = [x for x in new_point_set]
		self.label_set = [x for x in new_label_set]
		while True:
			new_label_set = [x for x in self.label_set]
			for x,idx in enumerate(self.point_set):
				new_label_set[idx] = getLabel(x)
			if new_label_set == self.label_set:
				break
			self.label_set = [x for x in new_label_set]
		return np.array(self.label_set)	 		
	def getLabel(self,point,distance='euclidean'):
		#point_set is a NxD numpy array. 
		#label_set is a Nx1 array of 0/1
		#point is a 1xD numpy array
		#Function returns a label for point
		if distance == 'euclidean':
			dist_mat = np.linalg.norm(np.array(self.data[self.point_set])-point,axis=1)
			return int(np.array(self.label_set)[dist_mat.argsort(axis=0)[:self.k]].sum()>self.k//2)
		else:
			raise NotImplementedError