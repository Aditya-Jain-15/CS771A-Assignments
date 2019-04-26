# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set

train_set = pd.read_fwf('kmeans_data.txt', header = None)

# K-means

# Feature Transformation (convert to polar coordinates)
def my_transform(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(r, phi)

def my_inv_transform(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

# Transform the original data
for i in range(np.shape(train_set)[0]):
  x = train_set[0][i]
  y = train_set[1][i]
  (train_set[0][i], train_set[1][i]) = my_transform(x, y)
  
# initialize means
mu_1 = train_set[0][0]
mu_2 = train_set[0][1]

z = np.empty(np.shape(train_set)[0])

while(True):
  
  # compute distance of each training data point from the means
  for i in range(np.shape(train_set)[0]):       
    
    dist1 = np.linalg.norm(train_set[0][i]-mu_1)    
    dist2 = np.linalg.norm(train_set[0][i]-mu_2)

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
      mu_1_new+=train_set[0][i]      

    else:
      cnt_2+=1
      mu_2_new+=train_set[0][i]      

  mu_1_new = mu_1_new/cnt_1
  mu_2_new = mu_2_new/cnt_2
  
  if(mu_1_new == mu_1 and mu_2_new == mu_2):
    break
  
  else:
    mu_1 = mu_1_new
    mu_2 = mu_2_new
    
# Inverse Transform back to the original data (for plotting)
for i in range(np.shape(train_set)[0]):
  r = train_set[0][i]
  phi = train_set[1][i]
  (train_set[0][i], train_set[1][i]) = my_inv_transform(r, phi)    

for i in range(len(z)):
  if z[i] == 1:
    plt.scatter(train_set[0][i], train_set[1][i], color = 'red')
  else:
    plt.scatter(train_set[0][i], train_set[1][i], color = 'green')
    
plt.show()
