import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm, linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import pickle
from utils import *
from kernel import *


class SVM_Classifier():
    def __init__(self,linear=True,kernel=linearKernel):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.linear = linear
        self.datadir= os.path.join('..','data')
        self.imgdir = os.path.join(self.datadir,'images')
        self.labeldir = os.path.join(self.datadir,'labels')
        # self.classifier = linear_model.SGDClassifier(max_iter=5000, verbose=1)
        self.classifier = svm.LinearSVC(max_iter=5000, class_weight = {1.0:1,-1.0:5.3}, verbose=1, C=0.1)  # -1 weight dec -> white inc
        self.feature_map = None
        self.max_iter = 100000
        self.mean = None
        self.std = None
        self.mean_feature = None
        self.std_feature = None
        if not linear:
            self.feature_map = Nystroem(gamma=1,random_state=42,n_components=10)

    def processData(self):
        imageList = os.listdir(self.imgdir)
        labelList = os.listdir(self.labeldir)
        imageList.sort()
        print(imageList)
        labelList.sort()
        print(labelList)
        self.mean, self.std = getNormalizationStatistics(self.imgdir)

        for i in range(len(imageList)):
            if i == 0:
                continue
            imagePath = os.path.join(self.imgdir,imageList[i])
            labelPath = os.path.join(self.labeldir,labelList[i])
            print(imagePath)
            feats = extractFeatureSVM(read_img(imagePath,gray=True),self.mean,self.std)
            labels = read_img(labelPath,gray=False)
            labels = labels.reshape(-1)/255.0
            if i==1:
                self.X_train = feats
                self.Y_train = labels
            else:
                self.X_train = np.vstack((self.X_train,feats))
                self.Y_train = np.hstack((self.Y_train,labels))

        # for i in [len(imageList)-2, len(imageList)-1]:
        for i in [0]:
            imagePath = os.path.join(self.imgdir,imageList[i])
            labelPath = os.path.join(self.labeldir,labelList[i])
            feats = extractFeatureSVM(read_img(imagePath,gray=True),self.mean,self.std)
            labels = read_img(labelPath,gray=False)
            labels = labels.reshape(-1)/255.0
            # if i==len(imageList)-2:
            if i==0:
                self.X_test = feats
                self.Y_test = labels
            else:
                self.X_test = np.vstack((self.X_test,feats))
                self.Y_test = np.hstack((self.Y_test,labels))


        #Normalize Training Data
        self.mean_feature = np.mean(self.X_train,axis=0)
        self.std_feature = np.std(self.X_train,axis=0)
        # self.X_train = (self.X_train-self.mean_feature)/self.std_feature
        # self.X_test = (self.X_test-self.mean_feature)/self.std_feature
        self.Y_train[self.Y_train==0]=-1
        self.Y_test[self.Y_test==0]=-1
        print(np.max(self.X_train,axis=0))
        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.Y_train.shape)
        print(self.Y_test.shape)
        print(np.sum(self.Y_test==0))


    def train(self):
        if not self.linear:
            self.X_train = self.feature_map.fit_transform(self.X_train)
            self.X_test = self.feature_map.fit_transform(self.X_test)

        self.classifier.fit(self.X_train,self.Y_train)


    def predict(self):
        Y_pred = self.classifier.predict(self.X_test)
        print(np.sum(Y_pred))
        print(confusion_matrix(self.Y_test, Y_pred))
        print(classification_report(self.Y_test, Y_pred))
        Y_pred[Y_pred==-1.0]=0
        generatedImage = Y_pred.reshape(605,700)
        cv2.imwrite('y_pred.png', 255.0*generatedImage)

    def predictImage(self,imagePath,linear=True):
        feats = extractFeatureSVM(read_img(imagePath,gray=True),self.mean,self.std)
        # feats = (feats - self.mean_feature)/self.std_feature
        # feats[:,1] = feats[:,1]*5
        # feats[:,2] = feats[:,2]*5
        print(np.max(feats,axis=0))
        if not linear:
            feats = self.feature_map.fit_transform(feats)
        pred_labels = np.array(self.classifier.predict(feats))
        print((np.sum(pred_labels)))
        # pred_labels[pred_labels==1] = 0
        pred_labels[pred_labels==-1.0] = 0
        print(pred_labels)
        generatedImage = pred_labels.reshape(605,700)
        cv2.imwrite('label.png', 255.0*generatedImage)




if __name__ == "__main__":
    svm_classifier= SVM_Classifier(linear=False)
    svm_classifier.processData()
    # c = svm_classifier.smo()    #
    svm_classifier.train()
    # # svm_classifier.predict()
    file = 'svm_model.pkl'
    with open(file,'wb') as f:
        pickle.dump(svm_classifier,f)

    svm_classifier.predict()

    svm_classifier.predictImage('../data/images/im0001.ppm',linear=False)



    # def predictImage():
