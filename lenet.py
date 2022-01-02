import pandas as pd
import numpy as np
import random
import math
from decimal import Decimal
import statistics


class ConvLayer:
    def __init__(self, inputs, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name, pca_init=False):
        self.filters = num_filters
        self.kernel = kernel_size
        self.channel = inputs_channel
        self.weights = np.zeros((self.kernel, self.kernel, self.channel, self.filters))
        self.bias = np.zeros((self.filters, 1))
        self.padding = padding
        self.stride = stride
        self.lr = learning_rate
        self.inputs = inputs
        self.name = name
        self.pca_init = pca_init
        self.momentum = 0.95
        self.weight_decay = 0.0001
        self.lr_decay = 0.0001
        self.weight_diff = 0
        self.bias_diff = 0

        if self.pca_init and self.name == 'c1':
            # randomly select a sample image to perform PCA
            # the # of selected eigenvectors match the # of convolutional filters
            #index = np.random.choice(self.inputs.shape[3], 1) # randomly selected image
            sample = self.inputs[:,:,:,0] 
            sizes = sample.shape
            val, vec = self.gen_pca_weights(sample, sizes)
            index_sorted = np.argsort(-val) # eigenvalue indices in descending order
            vec_sorted = vec[:,index_sorted] # sort eigenvectors
            vec_selected = vec_sorted[:,:self.filters] # select major eigenvectors by # filters
            
            w = int(self.inputs.shape[0]/2)-3
            h = int(self.inputs.shape[0]/2)-3
            for i in range(self.filters):
                vec_single_reshaped = vec_selected[:,i].reshape(sizes[0], sizes[1], sizes[2])
                self.weights[:,:,:,i] = vec_single_reshaped[w:w+self.kernel,h:h+self.kernel,:]
            self.weights = self.weights*np.sqrt(1./(self.filters))
        else: # random weights
            self.weights = np.random.randn(self.kernel, self.kernel, self.channel, self.filters)*np.sqrt(1./(self.kernel))
        
    def pca(self, sample, sizes):
        inputs = sample.reshape(sizes[0]*sizes[1]*sizes[2], 1) # reshape input to 1D array
        sigma = inputs*inputs.T/(inputs.shape[0]-1) # covariance matrix
        val, vec = np.linalg.eigh(sigma) # eigenvalues, eigenvectors
        return val, vec
        
    def gen_pca_weights(self, sample, sizes):
        pca_val, pca_vec = self.pca(sample, sizes)
        return pca_val, pca_vec
        
    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = w+2*size
        new_h = h+2*size
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out
    
    def forward_propagate(self, inputs):
        # k: kernel size, c: num channels, f: num filters, n: N/batch_size
        # w/h: original input sizes, ww/hh: reduced input sizes
        # weight size: (k, k, c, f)
        # bias size: (f)
        # input size: (w, w, c, n)
        # output size: (ww, hh, f, n)
        if inputs.shape[2] == 1: # 28x28 images
            w = inputs.shape[0]+2*self.padding
            h = inputs.shape[1]+2*self.padding
            c = inputs.shape[2]
            self.inputs = np.zeros((w, h, c))
        else:
            w = inputs.shape[0]
            h = inputs.shape[1]
            c = inputs.shape[2]
            self.inputs = np.zeros((w+2*self.padding, h+2*self.padding, c))
        for i in range(inputs.shape[2]): # zero paddingg
            self.inputs[:,:,i] = self.zero_padding(inputs[:,:,i], self.padding)
            
        ww = w-self.kernel+1
        hh = h-self.kernel+1
        feature_maps = np.zeros((ww, hh, self.filters))
        for w in range(ww):
            for h in range(hh):
                patch = self.inputs[w:w+self.kernel,h:h+self.kernel,:]
                for f in range(self.filters):
                    feature_maps[w,h,f] = np.sum(patch*self.weights[:,:,:,f])+self.bias[f]+self.bias[f]
    
        return feature_maps
        
    def back_propagate(self, dy):
        w, h, c = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        self.inputs = self.inputs/(w*h*c)
        
        w, h, f = dy.shape
        for i in range(w):
            for j in range(h):
                for k in range(f):
                    dw[:,:,:,k] += dy[i,j,k]*self.inputs[i:i+self.kernel,j:j+self.kernel,:]
                    dx[i:i+self.kernel,j:j+self.kernel,:] += dy[i,j,k]*self.weights[:,:,:,k]
                
        for i in range(f):
            db[i] = np.sum(dy[:,:,i])
        self.weight_diff = self.momentum*self.weight_diff + (1-self.momentum)*dw # weight decay
        self.bias_diff = self.momentum*self.bias_diff + (1-self.momentum)*db # bias decay
        dw = self.weight_diff
        db = self.bias_diff
        self.weights -= self.lr*dw + self.weight_decay*self.weights
        self.bias -= self.lr*db + self.weight_decay*self.bias
        self.lr *= (1-self.lr_decay) # learning decay
        return dx    
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self, weights):
        self.weights = weights
        
    def set_bias(self, bias):
        self.bias = bias
    
class MaxPoolLayer:
    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.stride = stride
        self.name = name
        self.inputs = ''
        
    def forward_propagate(self, inputs):
        #self.inputs = inputs
        self.inputs = (inputs-np.mean(inputs))/np.std(inputs)
        w, h, c = inputs.shape
        new_width = int((w-self.pool)/self.stride+1)
        new_height = int((h-self.pool)/self.stride+1)
        
        out = np.zeros((new_width, new_height, c))
        for i in range(int(w/self.stride)):
            for j in range(int(h/self.stride)):
                for k in range(c):
                    self.inputs = self.inputs.astype(np.float64)
                    out[i,j,k] = np.nanmax(self.inputs[i*self.stride:i*self.stride+self.pool, j*self.stride:j*self.stride+self.pool, k])       
        return out
        
    def back_propagate(self, dy):
        w, h, c = self.inputs.shape
        dx = np.zeros((w,h,c))
        for i in range(0, w, self.pool):
            for j in range(0, h, self.pool):
                for k in range(c):
                    max_arr = np.argmax(self.inputs[i:i+self.pool,j:j+self.pool,k])
                    pool_out = np.unravel_index(max_arr, (self.pool, self.pool))
                    dx[i+pool_out[0],j+pool_out[1],k] = dy[int(i/self.pool),int(j/self.pool),k]
        return dx
        
class Tanh:  
    def __init__(self):
        self.out = ''
        
    def forward_propagate(self, data):
        #self.out = np.tanh(data)
        self.out = (np.exp(data+10**(-7))-np.exp(-data+10**(-7)))/(np.exp(data+10**(-7))+np.exp(-data+10**(-7)))
        return self.out
        
    def back_propagate(self, dy):
        return 1-self.out**2
    
class ReLU():    
    def __init__(self):
        self.data = ''
        
    def forward_propagate(self, data):
        self.data = data
        output = data.copy()
        output = (output>0)*output
        return output

    def back_propagate(self, dy):
        output = dy.copy()
        if dy.shape != self.data.shape:
            output = output.T
        output = (self.data>0)*output
        return output
    
class Softmax:
    def __init__(self):
        self.out = ''
        
    def forward_propagate(self, data):
        exp = np.exp(data-np.max(data))
        self.out = exp/np.sum(exp)
        return self.out
    
    def back_propagate(self, dy):
        return self.out.T - dy.reshape(dy.shape[0],1)
        
class CrossEntropyLoss:
    def __init__(self, yhat, y):
        self.yhat = yhat
        self.y = y
        
    def evaluate(self):
        out = -np.sum(self.y*np.log(self.yhat+10**(-7)))/len(self.yhat)
        return out
    
    def gradient(self): 
        out = -self.y/(self.yhat+10**(-7))
        return out
        
class Flatten:
    def __init__(self):
        self.w = '' # width
        self.h = '' # height
        self.c = '' # channel
        
    def forward_propagate(self, data): 
        self.w, self.h, self.c = data.shape
        return data.reshape(self.w*self.h*self.c, 1)
    
    def back_propagate(self, dy):
        return dy.reshape(self.w, self.h, self.c) 
    
class FCLayer:
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = np.random.randn(num_inputs, num_outputs)*np.sqrt(2./num_inputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate
        self.name = name
        self.inputs = ''
        self.momentum = 0.95
        self.weight_decay = 0.0001
        self.lr_decay = 0.0001
        self.weight_diff = 0
        self.bias_diff = 0
        
    def forward_propagate(self, inputs):
        self.inputs = inputs/np.sqrt(inputs.shape[0])
        #out = (self.weights-np.mean(self.weights))/np.std(self.weights)
        out = (self.inputs.T@self.weights+self.bias.T).T
        return out
    
    def gradient(self): 
        return self.weights.T

    def back_propagate(self, dy):
        dy = dy/self.inputs.shape[0]**2
        grad_w = self.gradient()
        if dy.shape[1] != grad_w.shape[0]:
            dy = dy.T
        grad = dy@grad_w
        dw = self.inputs.T@grad.T
        db = np.sum(dy, keepdims=True)
        dx = dy@self.weights.T
        self.weight_diff = self.momentum*self.weight_diff + (1-self.momentum)*dw # weight decay
        self.bias_diff = self.momentum*self.bias_diff + (1-self.momentum)*db # bias decay
        dw = self.weight_diff
        db = self.bias_diff
        self.weights -= self.lr*dw + self.weight_decay*self.weights
        self.bias -= self.lr*db + self.weight_decay*self.bias
        self.lr *= (1-self.lr_decay) # learning decay
        return dx
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self, weights):
        self.weights = weights
        
    def set_bias(self, bias):
        self.bias = bias

