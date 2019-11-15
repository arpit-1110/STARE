import os
import sys
import numpy as np
from scipy.misc import imread, imresize
import cv2
from scipy.misc import imshow
import torch
from PIL import Image
# import warnings
# warnings.filterwarnings('error')



def read_img(path, gray=True, reshape=False, shape=(256,256)):
    if gray:
        img = imread(path)[:,:,1]
        if reshape:
            img = cv2.resize(img,shape,interpolation=cv2.INTER_NEAREST)

    else:
        img = imread(path)
        if reshape:
            img = cv2.resize(img,shape,interpolation=cv2.INTER_NEAREST)

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


def extractFeature(img,mean,std):
    img = normalizeImage(img,mean,std)
    img = clahe(img)
    img = adjustGamma(img)
    img = img/255
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

def adjustGamma(img,gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(np.array(img, dtype = np.uint8), table)

def clahe(img,clipLimit=2.0,tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    return clahe.apply(np.array(img,dtype=np.uint8))


def getNormalizationStatistics(img_path):
    images = []
    for img in os.listdir(img_path):
        images.append(read_img(os.path.join(img_path,img),gray=True))
    images = np.array(images)
    return np.mean(images),np.std(images)

def normalizeImage(img,mean,std):
    img = (img - mean)/std
    img = (img - np.min(img))/(np.max(img)-np.min(img))*255.0
    return img




if __name__ == "__main__":
    # img = read_img('../data/images/im0001.ppm', gray=True)
    mean,std = getNormalizationStatistics('../data/images')
    print(mean, std)
    img = read_img('../data/images/im0001.ppm',gray=True)
    imshow(img)
    img = normalizeImage(img,mean,std)
    img = clahe(img)
    print(img.shape)
    img = adjustGamma(img,1.2)
    print(np.max(img))

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gridsize = 40
    # clahe = cv2.createCLAHE(clipLimit=4,tileGridSize=(gridsize,gridsize))
    # img = clahe.apply(img)
    imshow(img)
    # img = cv2.GaussianBlur(img,(31,31),2)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(70,70))
    # back = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    # imshow(img)
    # imshow(back)
    # res = img-back
    # imshow(res)


    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # # imshow(lab_planes[0])
    # gridsize = 20
    # clahe = cv2.createCLAHE(clipLimit=4,tileGridSize=(gridsize,gridsize))
    # img = clahe.apply(img)
    # # lab_planes[0] = clahe.apply(lab_planes[0])
    # # lab_planes[1] = clahe.apply(lab_planes[1])
    # # lab_planes[2] = clahe.apply(lab_planes[2])
    # imshow(clahe.apply(img))
    # lab = cv2.merge(lab_planes)
    # # img = cv2.cvtColor(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),cv2.COLOR_RGB2GRAY)
    # # # img = clahe.apply(img)
    # # # print(img.shape)
    # imshow(img)
    # cv2.imshow('img',img)
    # F = delF(img)
    # print(F)
    # imshow(F)
