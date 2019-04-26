# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set
train_set = pd.read_fwf('kmeans_data.txt', header = None)

# Kernel K-means
gamma = 0.1

for cnt in range(10): # run 10 times
  Landmark_set = train_set.sample(n=1)

  # Extract Landmark based features using RBF Kernel
  psi = np.zeros(np.shape(train_set)[0])

  for i in range(np.shape(train_set)[0]):    
    x = train_set[0][i]
    y = train_set[1][i]
    pt = np.array([x, y])

    psi[i] = np.exp(-gamma*(np.linalg.norm(Landmark_set - pt)))

  # initialize means
  mu_1 = np.array([train_set[0][0], train_set[1][0]])
  mu_2 = np.array([train_set[0][1], train_set[1][1]])

  z = np.empty(np.shape(train_set)[0])

  while(True):

    # compute distance of each training data point from the means
    for i in range(np.shape(train_set)[0]):        

      psi_mu_1 = np.exp(-gamma*(np.linalg.norm(Landmark_set - mu_1)))
      psi_mu_2 = np.exp(-gamma*(np.linalg.norm(Landmark_set - mu_2)))

      dist1 = psi[i]**2 + psi_mu_1**2 - 2*psi[i]*psi_mu_1
      dist2 = psi[i]**2 + psi_mu_2**2 - 2*psi[i]*psi_mu_2

      # assign labels
      if dist1 < dist2:
        z[i] = 1
      else:
        z[i] = 2

    # recompute means
    mu_1_new = 0
    mu_2_new = 0
    cnt_1 = 0
    cnt_2 = 0

    for i in range(len(z)):
      if z[i] == 1:
        cnt_1+=1
        mu_1_new+= np.exp(-gamma*(np.linalg.norm(Landmark_set - train_set[0][i])))

      else:
        cnt_2+=1
        mu_2_new+=np.exp(-gamma*(np.linalg.norm(Landmark_set - train_set[0][i])))

    mu_1_new = mu_1_new/cnt_1
    mu_2_new = mu_2_new/cnt_2

    if(np.linalg.norm(mu_1_new - mu_1) and np.linalg.norm(mu_2_new - mu_2)):
      break

    else:
      mu_1 = mu_1_new
      mu_2 = mu_2_new

  # Plot
  fig = plt.figure(figsize=(10, 6))
    
  for i in range(len(z)):
    if z[i] == 1:
      plt.scatter(train_set[0][i], train_set[1][i], color = 'red')
    else:
      plt.scatter(train_set[0][i], train_set[1][i], color = 'green')

  plt.scatter(Landmark_set[0], Landmark_set[1], color = 'blue')
  plt.show()