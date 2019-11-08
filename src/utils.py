import os
import sys
import numpy as np
from scipy.misc import imread
import cv2
from scipy.misc import imshow
# import warnings
# warnings.filterwarnings('error')


def read_img(path, gray=True):
    if gray:
        img = imread(path, 'L')
    else:
        img = imread(path)
    return img


def delF(img):
    Fx, Fy = np.gradient(img)
    return np.sqrt(Fx**2 + Fy**2 + 1e-7)

def maxEigofHess(img):
    Fx, Fy = np.gradient(img)
    Fxx, Fxy = np.gradient(Fx)
    Fyy, _ = np.gradient(Fy)

    eig = (Fxx + Fyy + ((Fxx - Fyy)**2 + (2*Fxy)**2 + 1e-7)**0.5)/2.0
    return eig


def extractFeature(img):
    featImg = np.zeros((img.shape[0]*img.shape[1], 3))
    featImg[:, 0] = img.reshape(-1)
    featImg[:, 1] = delF(img).reshape(-1)
    featImg[:, 2] = maxEigofHess(img).reshape(-1)

    return featImg


def get_dataset(img_path, label_path):
    images = []
    for img in os.listdir(img_path):
        images.append(read_img(img_path + '/' + img))
    labels = []
    for label in os.listdir(label_path):
        labels.append(read_img(label_path + '/' + label))
    
    return np.array(images), np.array(labels)


# if __name__ == "__main__":
#     img = read_img('../data/images/im0001.ppm', gray=True)
#     # print(img.shape)
#     F = delF(img)
#     print(F)
#     imshow(F)