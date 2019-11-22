import keras
import tensorflow as tf
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
import keras.optimizers as opt
from keras.utils import to_categorical
from utils import extractFeature, read_img
import numpy as np
import cv2
import keras.backend as K

def myloss():
    def custom_loss(y_true, y_pred):
        true = K.variable(y_true)*K.variable(np.array([1, 100]))
        with tf.Session() as sess:
            print(true.eval())
        return K.sum(-true*(K.log(y_pred)))
    return custom_loss

def model():
    model = Sequential()
    # model.add(Flatten())
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
    optimizer = opt.SGD(lr=3e-3, momentum=0.9)
    model.compile(optimizer = optimizer, loss = myloss(), metrics = ['accuracy'])
    return model

if __name__ == "__main__":
    # d = 1
    np.random.seed(42)
    img = read_img("../data/images/im0001.ppm", True)
    featImg = extractFeature(img)
    label = read_img("../data/labels/im0001.ah.ppm", True)
    print(np.count_nonzero(label == 255), label.shape[0]*label.shape[1])
    label = label/255
    label = label.reshape(-1, 1)
    print(featImg[label.nonzero()[0], :])
    print(label[label.nonzero()[0], :])
    mean = featImg.mean(axis=0)
    
    label = to_categorical(label)
    loss = myloss()
    print('loss', label[0], loss(label[0], [0.1, 0.9]))
    exit()
    model = model()
    # ann_estimator = KerasRegressor(build_fn=model, epochs=5, batch_size=64, verbose=1)
    # boosted_ann = AdaBoostRegressor(base_estimator= ann_estimator, n_estimators=10)
    # boosted_ann.fit(featImg, label)
    # optimizer = opt.RMSprop(lr=0.02, rho=0.9)
    # model.compile(loss='categorical_crossentropy',
    #           optimizer=optimizer,
    #           metrics=['accuracy'],
    #           )
    cv2.imwrite('img.png', (featImg[:,0].reshape(605, 700)))
    cv2.imwrite('label.png', 255*(label[:,1].reshape(605, 700)))
    model.fit(featImg, label, epochs=2, verbose=1, batch_size=128)
    test_img = read_img("../data/images/im0235.ppm", True)
    featImg = extractFeature(test_img)
    label = read_img("../data/labels/im0235.ah.ppm", True)
    label = label/255
    label = label.reshape(-1, 1)
    label = to_categorical(label)

    res_img = model.predict(featImg)
    # imshow(img)
    print(res_img)
    # res_img = (res_img[:,1] > 0.4)*1
    # print(np.max(res_img.reshape(605, 700)))
    cv2.imwrite('test.png', 255*(res_img[:,1].reshape(605, 700)))
    # cv2.waitKey()
    scores = model.evaluate(featImg, label, verbose=0)
    print(scores[1] * 100)

