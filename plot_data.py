import pandas as pd
import numpy as np
import random
import math
import scipy.io
import os
from decimal import Decimal
import statistics



# plot CIFAR-10 sample
# data labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(5, 5, figsize = (10,10))
axes = axes.ravel()
for i in range(25):
    index = np.random.randint(0, img_train.shape[3]) 
    axes[i].imshow(img_train[:,:,:,index])
    label_index = int(img_train_y[index,:])
    axes[i].set_title(labels[label_index], fontsize = 8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.1)


# plot SVHN sample
fig, axes = plt.subplots(5, 5, figsize=(10,10))
axes = axes.ravel()
for i in range(25):
    index = np.random.randint(0, img_train.shape[3]) 
    axes[i].imshow(digit_train[:,:,:,index])
    label_index = int(digit_train_y[index,:])
    axes[i].set_title(label_index, fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.2)


fig, axes = plt.subplots(5, 5, figsize = (10,10))
axes = axes.ravel()
for i in range(25):
    index = np.random.randint(0, mnist_train.shape[3]) 
    axes[i].imshow(mnist_train[:,:,:,index], cmap="gray")
    label_index = int(mnist_train_y[index,:])
    axes[i].set_title(label_index, fontsize = 8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.2)


