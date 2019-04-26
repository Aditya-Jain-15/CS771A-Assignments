# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:45:32 2018

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

# 1. compute the mean of each seen class
for i in range(40):
    
    len = np.shape(X_seen[i])[0] # total no. of samples of class 'i'
    for j in range(len):
        mu[i] = mu[i] + X_seen[i][j]
    
    mu[i] = mu[i]/len

# 2. Compute the similarity (dot product based) of each unseen class 
#    with each of the seen classes.
    
sim = np.zeros((10, 40))
# sim[i][j] stores the similarity of class 'i' with class 'j'
    # 'i' denotes unseen class
    # 'j' denotes seen class
    
for i in range(10):
    for j in range(40):
        for k in range(85):
            sim[i][j] = sim[i][j] + class_attributes_unseen[i][k]*class_attributes_seen[j][k]


# 3. Normalize the similarity vector
for i in range(10):    
    sim[i] = sim[i]/(sum(sim[i]))


# 4. Compute the mean of each unseen class using a convex combination 
#    of means of seen classes.
    
for i in range(10):
    for j in range(40):
        mu[i+40] = mu[i+40] + sim[i][j]*mu[j]        


# 5. Apply prototype based classification

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

# 6. Compute classification accuracy
total_match = 0

for i in range(6180):
    if(y_pred[i]==Ytest[i]):
        total_match = total_match + 1

accuracy = total_match/6180 
print(accuracy)