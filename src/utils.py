import os
import sys
import numpy as np
from scipy.misc import imread
from scipy.misc import imshow


def read_img(path, gray):
    if gray:
        img = imread(path, 'L')
    else:
        img = imread(path)
    return img


def delF(img):
    Fx, Fy = np.gradient(img)
    return np.sqrt(Fx**2 + Fy**2)

def maxEigofHess(img):
    Fx, Fy = np.gradient(img)
    Fxx, Fxy = np.gradient(Fx)
    Fyy, _ = np.gradient(Fy)

    eig = (Fxx + Fyy + np.sqrt((Fxx - Fyy)**2 + (2*Fxy)**2))/2.0
    return eig


def extractFeature(img):
    featImg = np.zeros((3, img.shape[0], img.shape[1]))
    featImg[0, :, :] = img
    featImg[1, :, :] = delF(img)
    featImg[2, :, :] = maxEigofHess(img)

    return featImg



# if __name__ == "__main__":
#     img = read_img('../data/images/im0001.ppm', gray=True)
#     # print(img.shape)
#     F = delF(img)
#     imshow(F)