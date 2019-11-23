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
        self.classifier = svm.LinearSVC(max_iter=5000, class_weight = {1.0:1,-1.0:7.6}, verbose=1, C=0.1)  # -1 weight dec -> white inc
        self.feature_map = None
        self.C_pos = 0.14
        self.C_neg = 1.63
        self.kernel = kernel
        self.supportVectors = None
        self.supportVectorLabels = None
        self.tol = 0.001
        self.lagrange_multiplier_toi = 0.01
        self.max_iter = 100000
        self.alphas = None
        self.intercept = None
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
            feats = extractFeature(read_img(imagePath,gray=True),self.mean,self.std)
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
            feats = extractFeature(read_img(imagePath,gray=True),self.mean,self.std)
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


    def smo(self):  # Sequential Minimal Optimization
        N,D = self.X_train.shape
        alphas = np.zeros(N)
        grads = np.ones(N)

        iteration = 1
        while(True):

            print(iteration)
            #Compute Working Set
            y_grad = grads*self.Y_train
            index_pos = self.Y_train==1
            index_neg = self.Y_train==-1
            index_alpha_cpos = alphas>=self.C_pos
            index_alpha_cneg = alphas>=self.C_neg
            index_alpha_neg = alphas<=0

            invalid_max = np.logical_and(index_pos,index_alpha_cpos) + np.logical_and(index_neg,index_alpha_neg)
            y_grad_max = y_grad.copy()
            y_grad_max[invalid_max] = float('-inf')
            i = np.argmax(y_grad_max)

            invalid_min = np.logical_and(index_neg,index_alpha_cneg) + np.logical_and(index_pos,index_alpha_neg)
            y_grad_min = y_grad.copy()
            y_grad_min[invalid_min] = float('+inf')
            j = np.argmin(y_grad_min)


            stop_criterion = y_grad_max[i] - y_grad_min[j] < self.tol
            # print(y_grad_max[i] - y_grad_min[j])
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break

            ## Change code from here
            lambda_max_1 = (self.Y_train[i] == 1) * self.C_pos - self.Y_train[i] * alphas[i]
            lambda_max_2 = self.Y_train[j] * alphas[j] + (self.Y_train[j] == -1) * self.C_neg
            lambda_max = np.min([lambda_max_1, lambda_max_2])

            Ki = self.kernel(self.X_train, self.X_train[i,:]).reshape(-1)
            Kj = self.kernel(self.X_train, self.X_train[j,:]).reshape(-1)
            lambda_plus = (y_grad_max[i] - y_grad_min[j]) / (Ki[i] + Kj[j] - 2 * Ki[j])
            lambda_param = np.max([0, np.min([lambda_max, lambda_plus])])

            # Update gradient
            grads = grads + lambda_param * self.Y_train * (Kj - Ki)

            # Direction search update
            alphas[i] = alphas[i] + self.Y_train[i] * lambda_param
            alphas[j] = alphas[j] - self.Y_train[j] * lambda_param

            iteration += 1

            intercept = self._compute_intercept(alphas,y_grad)
            indices = alphas > self.lagrange_multiplier_toi
            print(np.sum(indices))

            # print(np.sum(indices == (self.Y_train==1)))


        self.alphas = alphas
        self.intercept = intercept


    def _compute_intercept(self, alphas, y_grad):
        index_pos = (self.Y_train == 1)
        index_neg = (self.Y_train == -1)
        indices = index_pos* (alphas < self.C_pos) * (alphas > 0)
        indices += index_neg*(alphas < self.C_neg) * (alphas > 0)
        return np.mean(y_grad[indices])


    ## Gram Matrix Computation won't work since it will give memory error.
    # def getGramMatrix(self):
    #     for i,x_i in enumerate(self.X_train):
    #         print(i)
    #         if i == 0:
    #             self.gramMatrix = self.kernel(self.X_train,self.X_train[i,:]).reshape(1,-1)
    #         else:
    #             self.gramMatrix = np.vstack([self.gramMatrix,self.kernel(self.X_train,self.X_train[i,:]).reshape(1,-1)])



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
        feats = extractFeature(read_img(imagePath,gray=True),self.mean,self.std)
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
    # with open(file,'rb') as g:
    #     svm_classifier = pickle.load(g)

    # # print(svm_classifier.X_train.shape)
    svm_classifier.predict()
    # imagePath = '../data/images/im0324.ppm'
    # feats = extractFeature(read_img(imagePath,gray=True))
    # pred_labels = np.array(svm_classifier.classifier.predict(feats))
    # print((np.sum(pred_labels)))
    # generatedImage = pred_labels.reshape(256,256)
    # cv2.imwrite('label.png', 255.0*generatedImage)

    svm_classifier.predictImage('../data/images/im0001.ppm',linear=False)



    # def predictImage():
