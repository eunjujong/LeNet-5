import pandas as pd
import numpy as np
import random
import math
import scipy.io
import os
from decimal import Decimal
import statistics


# samller samples
# SVHN dataset
indices = np.random.choice(digit_train.shape[3], 2000) # train:test=5:1
np.random.shuffle(indices)
X_train_svhn = digit_train[:,:,:,indices[:1500]]
y_train_svhn = digit_train_y[indices[:1500],:]
X_test_svhn = digit_train[:,:,:,indices[1500:]]
y_test_svhn = digit_train_y[indices[1500:],:]

# CIFAR-10 dataset
indices2 = np.random.choice(img_train.shape[3], 2000) # train:test=5:1
np.random.shuffle(indices2)
X_train_cifar = img_train[:,:,:,indices2[:1500]]
y_train_cifar = img_train_y[indices2[:1500],:]
X_test_cifar = img_train[:,:,:,indices2[1500:]]
y_test_cifar = img_train_y[indices2[1500:],:]

# MNIST dataset
indices3 = np.random.choice(mnist_train.shape[3], 2000) # train:test=5:1
np.random.shuffle(indices3)
X_train_mnist = mnist_train[:,:,:,indices3[:1500]]
y_train_mnist = mnist_train_y[indices3[:1500],:]
X_test_mnist = mnist_train[:,:,:,indices3[1500:]]
y_test_mnist = mnist_train_y[indices3[1500:],:]