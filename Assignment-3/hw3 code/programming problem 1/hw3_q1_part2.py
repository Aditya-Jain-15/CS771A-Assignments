# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set

train_set = pd.read_fwf('ridgetrain.txt', header = None)
test_set = pd.read_fwf('ridgetest.txt', header = None)

# Landmark Ridge

# Use RBF Kernel with gamma = 0.1, reg_param = 0.1

reg_param = 0.1
gamma = 0.1

X_train = train_set[0] # first column (single feature of training set)
y_train = train_set[1] # label for training set

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = test_set[0] # first column (single feature of test set)
y_test = test_set[1] # label for test set

X_test = np.array(X_test)
y_test = np.array(y_test)

L = [2, 5, 20, 50, 100]

# Do for each choice of L
for l in range(len(L)):
  
  # Choose L number of randomly chosen training points
  Landmark_set = train_set.sample(n=L[l])
  Landmark_X = np.array(Landmark_set[0])
  
  # Extract Landmark based features using RBF Kernel
  psi = np.zeros((np.shape(train_set)[0], L[l]))
    
  for i in range(np.shape(train_set)[0]):
    for j in range(L[l]):
      psi[i][j] = np.exp(-gamma*(Landmark_X[j] - X_train[i])**2)

  # Train using Linear Ridge Regression
  K = np.eye(np.shape(X_train)[0]) # Kernel Matrix (would be corrected in the next loop)

  for i in range(np.shape(X_train)[0]):
    for j in range(np.shape(X_train)[0]):
      K[i][j] = np.dot(psi[i], psi[j])
  
  I = np.eye(np.shape(X_train)[0]) # Identity Matrix
  alpha = np.linalg.inv(K + reg_param*I) * np.mat(y_train).transpose() # alpha vector
  
  # weight vector  
  w = np.zeros(L[l])
  for i in range(np.shape(X_train)[0]):
    w = w + alpha[i]*np.array(psi[i])
    
# predict on test set
  y_pred = np.zeros(np.shape(X_test)[0])

  for i in range(np.shape(X_test)[0]):
    s = np.zeros(L[l]) # denotes psi(x)
    for j in range(L[l]):
      s[j] = np.exp(-gamma*(Landmark_X[j] - X_test[i])**2)
    
    y_pred[i] = np.dot(w,s)

 # Plot
  X = np.array(X_test).reshape((np.shape(X_test)[0],)) # test set feature value
  y = np.array(y_test).reshape((np.shape(X_test)[0],)) # true labels
  y2 = np.array(y_pred).reshape((np.shape(X_test)[0],)) # predicted labels
  
  fig = plt.figure(figsize=(12, 6))
  
  ax1 = fig.add_subplot(1, 2, 1)
  
  ax1.scatter(X, y, color = 'blue', label = "true labels")
  plt.title("L = %i" %L[l])
  plt.xlabel("Test Set Feature Values")
  plt.ylabel("True Labels")
  plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
  
  ax2 = fig.add_subplot(1, 2, 2)
  
  ax2.scatter(X, y2, color = 'red', label = "predicted labels")
  plt.title("L = %i" %L[l])
  plt.xlabel("Test Set Feature Values")
  plt.ylabel("Predicted Labels") 
  plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
  
  fig2 = plt.figure(figsize=(10, 6))
  
  plt.scatter(X, y, color = 'blue', label = "predicted labels")
  plt.scatter(X, y2, color = 'red', label = "predicted labels")
  plt.title(" Both Plots Together: L = %i" %L[l])
    
  plt.show()

  # Calculate RMSE
  rmse = 0
  for i in range(np.shape(X_test)[0]):
    rmse+=(y_pred[i] - y_test[i])**2

  rmse/=np.shape(X_test)[0]
  print("rmse = ", rmse, '\n\n')