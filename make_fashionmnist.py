# -*- coding: utf-8 -*-

import os
import gzip
import numpy as np
import cv2




data_dir = "./data"


train = 'train'
val = 'val'
test = 'test'



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def save_imgs(data, label, save_dir='train'):
    print(data.shape, label.shape)#(60000, 784) (60000,)
    data_save_dir = os.path.join(data_dir, save_dir)
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    #make dir for each categray
    for i in range(10):
        cate_dir = os.path.join(data_save_dir, str(i))
        if not os.path.exists(cate_dir):
            os.mkdir(cate_dir)

    for i in range(len(data)):
        img = np.reshape(data[i], (28,28))
        img = cv2.resize(img, (224,224))
        gt = label[i]
        img_path = os.path.join(data_save_dir, str(gt), str(i)+".jpg")
        cv2.imwrite(img_path, img)








if __name__ == '__main__':

    X_train, y_train = load_mnist(data_dir, kind='train')
    X_test, y_test = load_mnist(data_dir, kind='t10k')

    X_val, y_val = X_train[59000:], y_train[59000:]
    X_train, y_train = X_train[:59000], y_train[:59000]

    save_imgs(X_train, y_train, save_dir=train)
    save_imgs(X_val, y_val, save_dir=val)
    save_imgs(X_test, y_test, save_dir=test)
