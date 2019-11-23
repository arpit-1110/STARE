import numpy as np
import queue

def clean_small_areas(I, area):

    labels = label_areas(I.astype(int))
    sizes = np.bincount(labels.ravel())

    sizes = sizes < area
    I[sizes[labels]] = 0

    return I



def label_areas(img):
	
	[n,m] = img.shape
	label_areass = np.zeros((n,m))
	curr = 1
	for i in range(n):
		for j in range(m):
			if img[i,j] == 0 or label_areass[i,j] != 0:
				continue
			else:
				Q = queue.Queue()
				Q.put([i,j])

				while Q.empty() == False:
					[x,y] = Q.get()
					label_areass[x,y] = curr

					if img[x+1,y]==1 and label_areass[x+1,y]==0:
						label_areass[x+1,y] = curr
						Q.put([x+1,y])
					if img[x-1,y]==1 and label_areass[x-1,y]==0:
						label_areass[x-1,y] = curr
						Q.put([x-1,y])
					if img[x,y+1]==1 and label_areass[x,y+1]==0:
						label_areass[x,y+1] = curr
						Q.put([x,y+1])
					if img[x,y-1]==1 and label_areass[x,y-1]==0:
						label_areass[x,y-1] = curr
						Q.put([x,y-1])

				curr+=1
	return label_areass.astype(int)