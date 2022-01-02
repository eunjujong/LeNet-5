import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy.io
import os
from decimal import Decimal
import statistics
import time
from cnn_train_test import *
from lenet import *
from subsample import *

if __name__ == '__main__':
    # SVHN dataset
    # random weight initialization
    # learning_rate = 0.001
    # batch_size = 32
    # epoch = 10
    cnn_reg_svhn = CNN(X_train_svhn, learning_rate = 0.001, pca=False)
    cnn_reg_svhn.train(X_train_svhn, y_train_svhn, batch_size=32, epoch=10)
    cnn_reg_svhn.test(X_test_svhn, y_test_svhn, epoch=10)

    # SVHN dataset
    # PCA weight initialization
    # learning_rate = 0.001
    # batch_size = 32
    # epoch = 10
    cnn_pca_svhn = CNN(X_train_svhn, learning_rate = 0.001, pca=True)
    cnn_pca_svhn.train(X_train_svhn, y_train_svhn, batch_size=32, epoch=10)
    cnn_pca_svhn.test(X_test_svhn, y_test_svhn, epoch=10)

    # CIFAR-10 dataset
    # random weight initialization
    # learning_rate = 0.001
    # batch_size = 32
    # epoch = 10
    cnn_reg_cifar = CNN(X_train_cifar, learning_rate = 0.001, pca=False)
    cnn_reg_cifar.train(X_train_cifar, y_train_cifar, batch_size=32, epoch=10)
    cnn_reg_cifar.test(X_test_cifar, y_test_cifar, epoch=10)

    # CIFAR-10 dataset
    # PCA weight initialization
    # learning_rate = 0.001
    # batch_size = 128
    # epoch = 10
    cnn_pca_cifar = CNN(X_train_cifar, learning_rate = 0.001, pca=True)
    cnn_pca_cifar.train(X_train_cifar, y_train_cifar, batch_size=32, epoch=10)
    cnn_pca_cifar.test(X_test_cifar, y_test_cifar, epoch=10)

    # MNIST dataset
    # random weight initialization
    # learning_rate = 0.0003
    # batch_size = 32
    # epoch = 10
    cnn_reg_mnist = CNN(X_train_mnist, learning_rate = 0.001, pca=False)
    cnn_reg_mnist.train(X_train_mnist, y_train_mnist, batch_size=32, epoch=10)
    cnn_reg_mnist.test(X_test_mnist, y_test_mnist, epoch=10)

    # MNIST dataset
    # PCA weight initialization
    # learning_rate = 0.0003
    # batch_size = 32
    # epoch = 10
    cnn_pca_mnist = CNN(X_train_mnist, learning_rate = 0.001, pca=True)
    cnn_pca_mnist.train(X_train_mnist, y_train_cifar, batch_size=32, epoch=10)
    cnn_pca_mnist.test(X_test_mnist, y_test_mnist, epoch=10)

    train_loss, test_loss = cnn_pca_mnist.get_loss()
    fig, ax = plt.subplots(figsize=(8,6))
    # training data
    ax.plot(train_loss, label='Train', color='blue')
    ax.set_xlabel('Epoch'); 
    ax.set_ylabel('Loss');
    ax.set_title('Cost Function of Regular CNN on SVHN')

    # testing data
    ax1 = ax.twinx()
    ax1.plot(test_loss, label='Test', color='red', linestyle='dashed')
    ax1.set_yticks([])
    fig.legend(ncol=1, borderaxespad=4);

    cnn_pca_mnist.plot_featuremap()
