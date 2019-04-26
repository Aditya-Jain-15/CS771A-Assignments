# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set

train_set = pd.read_fwf('ridgetrain.txt', header = None)
test_set = pd.read_fwf('ridgetest.txt', header = None)

# Kernel Ridge Regression

# Use RBF Kernel with gamma = 0.1

# Do for each choice of reg_param
reg_param = [0.1, 1, 10, 100]

X_train = train_set[0] # first column (single feature of training set)
y_train = train_set[1] # label for training set

# convert to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = test_set[0] # first column (single feature of test set)
y_test = test_set[1] # label for test set

# convert to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

# Take transpose to match with the original dimensions
X_train = np.matrix(X_train).transpose()
y_train = np.matrix(y_train).transpose()

X_test = np.matrix(X_test).transpose()
y_test = np.matrix(y_test).transpose()

I = np.eye(np.shape(X_train)[0]) # Identity Matrix
gamma = 0.1

K = np.eye(np.shape(X_train)[0]) # Kernel Matrix (would be corrected in the next loop)

for i in range(np.shape(X_train)[0]):
  for j in range(np.shape(X_train)[0]):
    K[i][j] = np.exp(-gamma*(X_train[i]-X_train[j])**2)

# 1. train the model

for l in range(len(reg_param)):
  alpha = np.linalg.inv(K + reg_param[l] * I) * y_train # alpha vector

  # 2. Make predictions
  y_pred = np.zeros(np.shape(X_test)[0]) # y_pred stores the prediction of the model for the test set

  for j in range(np.shape(X_test)[0]):
    for k in range(np.shape(X_train)[0]):
      y_pred[j]+=alpha[k]*np.exp(-gamma*(X_train[k]-X_test[j])**2)

  y_pred = np.matrix(y_pred).transpose() # Make of suitable dimension

  # 3. Plot
  X = np.array(X_test).reshape((np.shape(X_test)[0],)) # test set feature value
  y = np.array(y_test).reshape((np.shape(X_test)[0],)) # true labels
  y2 = np.array(y_pred).reshape((np.shape(X_test)[0],)) # predicted labels

  
  fig = plt.figure(figsize=(12, 6))
  
  ax1 = fig.add_subplot(1, 2, 1)
  
  ax1.scatter(X, y, color = 'blue', label = "true labels")
  plt.title("lambda = %f" %reg_param[l])
  plt.xlabel("Test Set Feature Values")
  plt.ylabel("True Labels")
  plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
  
  ax2 = fig.add_subplot(1, 2, 2)
  
  ax2.scatter(X, y2, color = 'red', label = "predicted labels")
  plt.title("lambda = %f" %reg_param[l])
  plt.xlabel("Test Set Feature Values")
  plt.ylabel("Predicted Labels") 
  plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
  
  fig2 = plt.figure(figsize=(10, 6))
  
  plt.scatter(X, y, color = 'blue', label = "predicted labels")
  plt.scatter(X, y2, color = 'red', label = "predicted labels")
  plt.title(" Both Plots Together: lambda = %f" %reg_param[l])
  
  plt.show()

  # 4. Calculate RMSE
  rmse = 0
  for i in range(np.shape(X_test)[0]):
    rmse+=(y_pred[i] - y_test[i])**2

  rmse/=np.shape(X_test)[0]
  rmse = np.array(np.sqrt(rmse))
  rmse = rmse[0][0]
  
  print("rmse = ", rmse, '\n\n')