# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:01:18 2018

@author: ADITYA JAIN
"""

# importing libraries
import numpy as np

# importing data sets
class_attributes_seen=np.load('class_attributes_seen.npy')
class_attributes_unseen=np.load('class_attributes_unseen.npy')

Xtest=np.load('Xtest.npy')
Ytest=np.load('Ytest.npy') # ground truth labels of test data

X_seen=np.load('X_seen.npy', encoding = 'latin1')

# Procedure
mu = np.zeros((50, 4096)) # mu[i] stores the mean of class 'i'

# 1. Compute the mean of each seen class
for i in range(40):
    
    len = np.shape(X_seen[i])[0] # total no. of samples of class 'i'
    for j in range(len):
        mu[i] = mu[i] + X_seen[i][j]
    
    mu[i] = mu[i]/len

# 2. Learn the multi-output regression model with 
     # input -> class attribute vector 
     # output -> class mean vector
 
X_train = class_attributes_seen
y_train = mu[:40] 

W = np.zeros((np.shape(X_train)[1], np.shape(y_train)[1]))

X = np.mat(X_train)
y = np.mat(y_train)
I = np.eye(np.shape(X)[1])

reg_param = np.array([0.01, 0.1, 1, 10, 20, 50, 100])
accuracy = [] # stores the accuracy corresponding to each value of 'lambda'

for i in range(np.size(reg_param)):
    W = np.linalg.inv(X.transpose()*X + reg_param[i]*I)*X.transpose()*y
    
    # Apply the learned regression model to compute the mean of each
    # unseen class
    
    for i in range(10):
        mu[i+40] = np.reshape((class_attributes_unseen[i]*W).transpose(), (4096,))
    
    # Apply prototype based classification    
    y_pred = [] # stores the predictions of the test data
    
    for i in range(np.shape(Xtest)[0]):
            
        # compute distance of X_test[i] from the mean of all 10 unseen classes
        # choose the class with the least distance as the prediction
        
        min_dist = float("inf")
        predicted_class = -1
        
        for j in range(10): # the predicted class can only be from unseen class
            dist = 0
            dist = dist + np.linalg.norm(Xtest[i] - mu[j+40])
            
            if dist < min_dist:
                min_dist = dist
                predicted_class = j+1
        
        y_pred.append(predicted_class)
    
    # Compute classification accuracy
    total_match = 0
    
    for i in range(6180):
        if(y_pred[i]==Ytest[i]):
            total_match = total_match + 1
    
    accuracy.append(total_match/6180)

# Results
for i in range(np.array(accuracy).shape[0]):
    print("lambda =", reg_param[i], "->", accuracy[i])    