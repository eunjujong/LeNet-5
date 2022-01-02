import pandas as pd
import numpy as np
import random
import math
import scipy.io
import os
from decimal import Decimal
import statistics



# load Street View House Numbers (SVHN) Dataset
mat_train = scipy.io.loadmat('data/train_32x32.mat') 
mat_test = scipy.io.loadmat('data/test_32x32.mat') 
digit_train = mat_train['X']
digit_train_y = mat_train['y']
digit_test = mat_test['X']
digit_test_y = mat_test['y']



# load CIFAR-10 dataset
path, dirs, files = next(os.walk('./Data/cifar-10-batches-mat'))

img_train = []
img_train_y = []
for i in range(len(files)):
    if 'data' in files[i]:
        temp = scipy.io.loadmat(path+'/'+files[i])
        img_train.append(temp['data'].T.reshape(32, 32, 3, len(temp['data'])))
        img_train_y.append(temp['labels'])
    elif 'test' in files[i]:       
        temp = scipy.io.loadmat(path+'/'+files[i]) 
        img_test = temp['data'].T.reshape(32, 32, 3, len(temp['data']))
        img_test_y = temp['labels']
        
img_train = np.concatenate(img_train, axis=3)
img_train_y = np.concatenate(img_train_y, axis=0)


# convert to Fortran order
for i in range(img_train.shape[3]):
    img_train[:,:,:,i] = img_train[:,:,:,i].reshape(3,32,32).transpose(1,2,0)
for i in range(img_test.shape[3]):
    img_test[:,:,:,i] = img_test[:,:,:,i].reshape(3,32,32).transpose(1,2,0)

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    f.read(16)
    l.read(8)
    images = []
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


convert('./Data/train-images-idx3-ubyte', './Data/train-labels-idx1-ubyte', 'mnist_train.csv', 60000)
convert('./Data/t10k-images-idx3-ubyte', './Data/t10k-labels-idx1-ubyte', 'mnist_test.csv', 10000)



# load MNIST dataset
mnist_train = pd.read_csv('./Data/mnist_train.csv', header=None, index_col=False)
mnist_test = pd.read_csv('./Data/mnist_test.csv', header=None, index_col=False)
# separate labels
mnist_train_y = mnist_train[0]
mnist_test_y = mnist_test[0]
mnist_train = mnist_train.iloc[:,1:]
mnist_test = mnist_test.iloc[:,1:]


# reshape datasets
mnist_train = np.array(mnist_train).T.reshape(28, 28, 1, 60000)
mnist_train_y = np.array(mnist_train_y).reshape(60000,1)
mnist_test = np.array(mnist_test).T.reshape(28, 28, 1, 10000)
mnist_test_y = np.array(mnist_test_y).reshape(10000,1)


print('SVHN data training size:', digit_train.shape)
print('SVHN data test size:', digit_test.shape)
print('CIFAR data training size:', img_train.shape)
print('CIFAR data test size:', img_test.shape)
print('MNIST data training size:', mnist_train.shape)
print('MNIST data test size:', mnist_test.shape)


