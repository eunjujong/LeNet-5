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

class CNN:
    def __init__(self, train_x, learning_rate, pca):
        """
        reference: https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
        Lenet-5 Architecture
        
        input: 32x32x3
        conv1: (5x5x6) w/ stride 1, padding 2 -> 28x28x6 {(32-5+2x2)/1+1}
        maxpool2: (2x2) w/ stride 2 -> 14x14x6 {(28-2)/2+1}
        conv3: (5x5x16) w/ stride 1, padding 0 -> 10x10x16 {(14-5)/1+1}
        maxpool4: (2x2) w/ stride 2 -> 5x5x16 {(10-2)/2+1}
        conv5: (5x5x120) w/ stride 1, padding 0 -> 1x1x120 {(5-5)/1+1}
        fc6: 120 -> 84
        fc7: 84 -> 10
        softmax: 10 -> 10
        """
        self.train_x = train_x
        self.layers = []
        self.layers.append(ConvLayer(train_x, inputs_channel=train_x.shape[2], num_filters=6, kernel_size=5, padding=2, stride=1, learning_rate=learning_rate, name='c1', pca_init=pca))
        #self.layers.append(Tanh())
        self.layers.append(ReLU())
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2, name='s2'))
        self.layers.append(ConvLayer(train_x, inputs_channel=6, num_filters=16, kernel_size=5, padding=0, stride=1, learning_rate=learning_rate, name='c3', pca_init=pca))
        #self.layers.append(Tanh())
        self.layers.append(ReLU())
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2, name='s4'))
        self.layers.append(ConvLayer(train_x, inputs_channel=16, num_filters=120, kernel_size=5, padding=0, stride=1, learning_rate=learning_rate, name='c5', pca_init=pca))
        #self.layers.append(Tanh())
        self.layers.append(ReLU())
        self.layers.append(Flatten())
        self.layers.append(FCLayer(num_inputs=120, num_outputs=84, learning_rate=learning_rate, name='f6'))
        #self.layers.append(Tanh())
        self.layers.append(ReLU())
        self.layers.append(FCLayer(num_inputs=84, num_outputs=10, learning_rate=learning_rate, name='f7'))
        self.layers.append(Softmax())
        
        self.num_layers = len(self.layers)
        self.trainWeights = []
        self.trainBias = []
        self.train_acc_tracker = []
        self.train_loss_tracker = []
        self.test_acc_tracker = []
        self.test_loss_tracker = []
        
    def train(self, train_x, train_lab, batch_size, epoch):
        print('\nTraining\n')
        #train_x = train_x/255.0
        train_x = (train_x-np.mean(train_x))/np.std(train_x) # standardize data
        
        start_time = time.time()
        for i in range(epoch): # train
            train_acc = 0
            train_loss = 0
            batch_count = 0
            for j in range(0, train_x.shape[3], batch_size): 
                if j+batch_size < train_x.shape[3]: # splitting data
                    data = train_x[:,:,:,j:j+batch_size]
                    label = train_lab[j:j+batch_size,:]
                else:
                    data = train_x[:,:,:,j:train_x.shape[3]]
                    label = train_lab[j:train_lab.shape[0],:]
                               
                for b in range(data.shape[3]): # iterate over images
                    x = data[:,:,:,b]
                    y = label[b,:]
                    for l in range(self.num_layers):
                        output = self.layers[l].forward_propagate(x)
                        x = output
                        
                    cross_entropy = CrossEntropyLoss(output, y)
                    train_loss += cross_entropy.evaluate()
                    if np.argmax(output) == y:
                        train_acc += 1
                    #print(np.argmax(output))
                    # backpropagate
                    dy = y
                    for l in range(self.num_layers-1, -1, -1):
                        dout = self.layers[l].back_propagate(dy)
                        dy = dout
                
                batch_count += 1
            
            # track losses and accuracies
            batch_acc = float(train_acc)/float(train_x.shape[3])
            batch_loss = float(train_loss)/float(train_x.shape[3])
            self.train_acc_tracker.append(batch_acc)
            self.train_loss_tracker.append(batch_loss)
            print('Epoch: {:d}/{:d} - Batch Completed: {:d} - Loss: {:.4f} - Acc: {:.4f}'.format(i+1, epoch, batch_count, batch_loss, batch_acc))        
        
        end_time = time.time()   
        # preserve weights and biases
        for k in range(self.num_layers):
            if hasattr(self.layers[k], 'name'):
                if self.layers[k].name in ['c1', 'c3', 'c5', 'f6', 'f7']:
                    weights = self.layers[k].get_weights()
                    bias = self.layers[k].get_bias()
                    self.trainWeights.append(weights)
                    self.trainBias.append(bias)
        
        # final training accuracy
        print('\nTrain Size: {:d}\nTrainining Acc: {:.10f}'.format(train_x.shape[3], float(train_acc)/float(train_x.shape[3])))
        print('Training Loss: {:.10f}\n'.format(float(train_loss)/float(train_x.shape[3])))
        print('Training Time: {:.4f}s'.format(end_time-start_time))
        print('------------------------------------------------------------------------------\n')
        
    def test(self, test_x, test_lab, epoch):
        print('\nTesting\n')
        #test_x = test_x/255.0
        test_x = (test_x-np.mean(test_x))/np.std(test_x) # standardize data
        
        i = 0
        start_time = time.time()
        for k in range(self.num_layers):
            if hasattr(self.layers[k], 'name'):
                if self.layers[k].name in ['c1', 'c3', 'c5', 'f6', 'f7']:
                    weights = self.trainWeights[i]
                    bias = self.trainBias[i]
                    self.layers[k].set_weights(weights)
                    self.layers[k].set_bias(bias)
                    i += 1
        
        test_acc_tracker = []
        test_loss_tracker = []
        for i in range(epoch): 
            test_acc = 0
            test_loss = 0
            for j in range(test_x.shape[3]): # iterate over images
                x = test_x[:,:,:,j]
                y = test_lab[j,:]
                for l in range(self.num_layers):
                    output = self.layers[l].forward_propagate(x)
                    x = output
                        
                cross_entropy = CrossEntropyLoss(output, y)
                test_loss += cross_entropy.evaluate()
                if np.argmax(output) == y:
                    test_acc += 1
                
                #print(np.argmax(output))
                # backpropagate
                dy = y
                for l in range(self.num_layers-1, -1, -1):
                    dout = self.layers[l].back_propagate(dy)
                    dy = dout

            # track losses and accuracies
            acc = float(test_acc)/float(test_x.shape[3])
            loss = float(test_loss)/float(test_x.shape[3])
            self.test_acc_tracker.append(acc)
            self.test_loss_tracker.append(loss)
            print('Epoch: {:d}/{:d} - Loss: {:.10f} - Acc: {:.10f}'.format(i+1, epoch, loss, acc))
        
        end_time = time.time()
        
        # final testing accuracy
        print('\nTest Size: {:d}\nTesting Acc: {:.10f}'.format(test_x.shape[3], acc))
        print('Testing Loss: {:.10f}\n'.format(loss))
        print('Testing Time: {:.4f}s'.format(end_time-start_time))
        
    def get_accuracy(self):
        return self.train_acc_tracker, self.test_acc_tracker
    
    def get_loss(self):
        return self.train_loss_tracker, self.test_loss_tracker
    
    def plot_accuracy(self):
        train_acc, test_acc = cnn_temp.get_acc()
        fig, ax = plt.subplots(figsize=(8,6))
        # training data
        ax.plot(train_acc, label='Train', color='blue')
        ax.set_xlabel('Epoch'); 
        ax.set_ylabel('Accuracy');
        ax.set_title('Training and Testing Accuracies')

        # testing data
        ax1 = ax.twinx()
        ax1.plot(test_acc, label='Test', color='red', linestyle='dashed')
        ax1.set_yticks([])
        fig.legend(ncol=1, borderaxespad=4)
        
    def plot_loss(self):
        train_loss, test_loss = cnn_temp.get_loss()
        fig, ax = plt.subplots(figsize=(8,6))
        # training data
        ax.plot(train_loss, label='Train', color='blue')
        ax.set_xlabel('Epoch'); 
        ax.set_ylabel('Loss');
        ax.set_title('Training and Testing Losses')

        # testing data
        ax1 = ax.twinx()
        ax1.plot(test_loss, label='Test', color='red', linestyle='dashed')
        ax1.set_yticks([])
        fig.legend(ncol=1, borderaxespad=4)
    
    def plot_featuremap(self):
        output = self.layers[0].forward_propagate(self.train_x[:,:,:,0])
        fig, axes = plt.subplots(2, 3, figsize=(6,4))
        axes = axes.ravel()
        for i in range(output.shape[2]): 
            axes[i].imshow(output[:,:,i])
            axes[i].axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

