
# Lenet-5 (LeCun et al., 1998a)

This Lenet-5 implementation in python is a slightly modified version of the original LeNet-5 (LeCun et al., 1998a).  

The implementation has been tested on the Jupiter Notebook. 

The data to be used are the CIFAR-10 and the Street View Housing Numbers (SVHN). These dataset are loaded from the local machine. You need to change the path to where your datasets are located for loading and conversion. 

There are 6 classes that account for each layer of the CNN: convolution, Max Pool, ReLU, Softmax, Flatten, and Fully Connected. 

A CNN class accumulates the layer classes and construct a network based on predefined parameters. Modification on these parameters can be made directly from the CNN class. A 'train' and a 'test' methods from the CNN class can be called for training and testing with additional hyper parameter settings such as batch size, epoch, and learning rate. These methods outputs accuracy per epoch as well as run time. 

The regular CNN and the CNN with PCA are operated on the same CNN class. The only modification to be made for initialing the PCA training is to set the 'pca' to True when instantiating the CNN object. 

The following architecture assumes that the image has been extracted and resized to a 4D structure of (width, height, num channels, sample size). 

Built-in libraries are imported in the Notebook. (pandas, numpy, random, math, matplotlib.pyplot, scipy.io, os, decimal, statistics, time)
No additional libraries are needed to run the model.  


# Data Sources
http://ufldl.stanford.edu/housenumbers/
https://www.cs.toronto.edu/~kriz/cifar.html
http://yann.lecun.com/exdb/mnist/


