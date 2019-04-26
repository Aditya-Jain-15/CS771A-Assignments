# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# importing data set
train_set = pickle.load(open('facedata.pkl', 'rb'))

# np.shape(train_set['X']) # (165, 4096) -> 165 images (15 persons), each image 64*64

# implement simplified version of PPCA

N = np.shape(train_set['X'])[0]
D = np.shape(train_set['X'])[1]
X = np.matrix(train_set['X'])

n_iters = 100
K = [10, 20, 30, 40, 50, 100]

for k in K:
  
  # 1. Learn the model
  
    # Initialize W as W_new
  W_new = np.matrix(np.random.randn(D, k))
  W_new_transposed = W_new.transpose()

  for i in range(n_iters):

    Z_new_transposed = np.linalg.inv(W_new_transposed*W_new)*W_new_transposed*(X.transpose())
    Z_new = Z_new_transposed.transpose()

    W_new_transposed = np.linalg.inv(Z_new.transpose()* Z_new)*Z_new.transpose()*X
    W_new = W_new_transposed.transpose()
    # debugger:  print(np.linalg.norm(X - Z_new*W_new.transpose()))

  # 2. 

  # compute mu
  mu = np.zeros((D, 1))

  for i in range(N):
    mu = mu + X[i].transpose()

  mu = mu/N

  # Reconstruct first 5 images and plot along with the original images for comparison  
  fig = plt.figure(figsize=(25, 45))
  for i in range(5):
    X_cap = W_new*Z_new[i].transpose() # X_cap = reconstructed image

    # normalize X_cap
    mx = max(X_cap)
    mn = min(X_cap)
    X_cap = (X_cap - mn)/(mx - mn)*255
    
    # plot the original image    
    
    ax1 = fig.add_subplot(5, 2, 2*i+1)
    if(i==0):
        plt.title("original images: K = %i" %k)
      
    plt.imshow((X[i].reshape(64,64)).transpose(), cmap='gray')
    plt.grid("off")
    
    # plot the reconstructed image
    ax1 = fig.add_subplot(5, 2, 2*i+2)
    if(i==0):
        plt.title("reconstructed images: K = %i" %k)

    plt.imshow((X_cap.reshape(64,64)).transpose(), cmap='gray') 
    plt.grid("off")    
    
  plt.show()        
  # show 10 basis images  
  fig2 = plt.figure(figsize=(16, 32))
        
  for i in range(10):            
    ax1 = fig2.add_subplot(1, 10, i+1)
    plt.title("Base Img %i" %(i+1))
    plt.imshow((W_new[:, i].reshape(64,64)).transpose(), cmap='gray')    
    plt.grid("off")
    
  plt.show()

