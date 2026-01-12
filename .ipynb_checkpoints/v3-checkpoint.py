#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# In[2]:


X = np.load("data/x_train.npy") #[:3000]
Y = np.load("data/y_train.npy") #[:3000]

X_test = np.load("data/x_test.npy")
Y_test = np.load("data/y_test.npy")


# In[3]:


X = np.hstack((X, np.ones((X.shape[0], 1))))
X = X + 0.00000001


X_mean = X.mean(axis=0)

X_std = X.std(axis=0)

X = (X - X_mean) / X_std

# Add bias term to the input data X
 #[:1000]
#Y = Y[:1000]
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))


# In[4]:


def custom_onehot(Y):
    c = 10
    Y_new = np.zeros((Y.shape[0], 10))
    for i in range(Y.shape[0]):
        new = np.zeros((1, c))
       # if(Y[i] > 0):
        #    new[0][Y[i]-1] = 1
        #else:
        new[0][Y[i]] = 1
        Y_new[i] = new
    return Y_new

Y = custom_onehot(Y)


# In[5]:


print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


# In[6]:


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    """
    Numerically stable softmax for vectors or matrices.
    - If x is 1D: returns a 1D probability vector.
    - If x is 2D: applies softmax row-wise.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(s):
    """
    Derivative of softmax for each vector independently.
    s must be the softmax output.
    Returns the Jacobian for each vector.
    - If s is shape (n,): returns (n, n)
    - If s is shape (batch, n): returns (batch, n, n)
    """
    if s.ndim == 1:
        # single vector
        return np.diag(s) - np.outer(s, s)
    else:
        # batch of vectors
        batch = s.shape[0]
        n = s.shape[1]
        J = np.zeros((batch, n, n))
        for i in range(batch):
            si = s[i]
            J[i] = np.diag(si) - np.outer(si, si)
        return J


# In[7]:


LEARNING_RATE = 0.007
ITERATIONS = 100
FIRST_LAYER_NEURONS = 64
SECOND_LAYER_NEURONS = 128

m = X.shape[0]
i = X.shape[1]

# Initialize weights
theta_1 = np.random.rand(i, FIRST_LAYER_NEURONS).astype("float64")*0.01
theta_2 = np.random.rand(FIRST_LAYER_NEURONS + 1, SECOND_LAYER_NEURONS).astype("float64")*0.01
theta_out = np.random.rand(SECOND_LAYER_NEURONS + 1, 10).astype("float64")*0.01


# In[8]:


def forward_pass(t1, t2, t3, return_az_arrays = False):
    z1 = X.dot(t1)
    a1 = relu(z1)

    a1 = np.hstack((a1, np.ones((a1.shape[0], 1))))


    z2 = a1.dot(t2)
    a2 = relu(z2)

    a2 = np.hstack((a2, np.ones((a2.shape[0], 1))))

    z3 = a2.dot(t3)
    y_hat = softmax(z3)

    if (return_az_arrays): return y_hat, a1, a2, z1, z2
    return y_hat


# In[9]:


print(forward_pass(theta_1, theta_2, theta_out)[:4])
print(Y[1])


# In[10]:


def backprop(t1, t2, t3, X, Y):

    for i in range(ITERATIONS):
        print(f"iteration {i} starting...")
        y_hat, a1, a2, z1, z2 = forward_pass(t1, t2, t3, True)

        dz3 = y_hat - Y

        dt3 = a2.T.dot(dz3)
        db3 = dz3
        np.append(dt3, db3)

        da2 = dz3.dot(t3[:-1].T)
        dz2 = da2 * relu_derivative(z2)

        dt2 = a1.T.dot(dz2)
        db2 = dz2
        np.append(dt3, db3)

        da1 = dz2.dot(t2[:-1].T)
        dz1 = da1 * relu_derivative(z1)

        dt1 = X.T.dot(dz1)
        db1 = dz1
        np.append(dt1, db1)

        t1 = t1 - LEARNING_RATE * dt1
        t2 = t2 - LEARNING_RATE * dt2
        t3 = t3 - LEARNING_RATE * dt3
        print(f"iteration {i} finished.")

    return t1, t2, t3


# In[11]:


print(f"Hey there 1: {forward_pass(theta_1, theta_2, theta_out)[:5]}")
theta_1, theta_2, theta_out = backprop(theta_1, theta_2, theta_out, X, Y)
print(f"Hey there 2: {forward_pass(theta_1, theta_2, theta_out)[:10]}")
print(f"Hey there 3: {Y[:5]}")


# In[ ]:




