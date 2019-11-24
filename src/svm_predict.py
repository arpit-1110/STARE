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

if __name__ == "__main__":
    file = 'svm_model.pkl'
    with open(file,'rb') as f:
        svm_classifier = pickle.load(f)

    svm_classifier.predict()
