import numpy as np

'''The kernel functions can work in such a way that the first element is a 2d array
    of size n*d and the second element is a 1d array of features
'''
def linearKernel(x,y):
    return x@y.reshape(-1,1)

def polynomialKernel(x, y, degree=3):
    return (1+x@y.reshape(-1,1))**degree

def gaussianKernel(x, y, sigma=1.0):
    return np.exp(-(linalg.norm((x-y,reshape(1,-1)),axis=1))**2/(2*(sigma**2)))
