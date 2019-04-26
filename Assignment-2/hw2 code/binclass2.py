
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data sets
df = pd.read_csv('binclassv2.txt', header=None)

# Procedure
    
# 1. Split data set into 2 parts, 
    # first part -> label +1, second part -> label -1
    
first_part = df.loc[df[2] == 1]
second_part = df.loc[df[2] == -1]

# 2. find mu_+ (aka mu_pos) and mu_- (aka mu_neg)

# mu_+
mu_pos = [0, 0]

mu_pos[0] = sum(first_part[0])
mu_pos[1] = sum(first_part[1])

mu_pos = np.array(mu_pos)

N_pos = len(first_part)
mu_pos = mu_pos/N_pos

# mu_-
mu_neg = [0, 0]

mu_neg[0] = sum(second_part[0])
mu_neg[1] = sum(second_part[1])

mu_neg = np.array(mu_neg)

N_neg = len(second_part)
mu_neg = mu_neg/N_neg

# 1. Generative Classification with Gaussian Class-Conditionals (different covariances)

# find (sigma_+)^2 and (sigma_-)^2

# (sigma_+)^2
X_diff_1_pos = np.array(first_part[0] - mu_pos[0]) # first coordinate
X_diff_2_pos = np.array(first_part[1] - mu_pos[1]) # second coordinate

X_diff_1_pos = X_diff_1_pos**2
X_diff_2_pos = X_diff_2_pos**2

sigma_pos_squared = sum(X_diff_1_pos) + sum(X_diff_2_pos)
sigma_pos_squared = sigma_pos_squared/(2*N_pos)

# (sigma_-)^2
X_diff_1_neg = np.array(second_part[0] - mu_neg[0]) # first coordinate
X_diff_2_neg = np.array(second_part[1] - mu_neg[1]) # second coordinate

X_diff_1_neg = X_diff_1_neg**2
X_diff_2_neg = X_diff_2_neg**2

sigma_neg_squared = sum(X_diff_1_neg) + sum(X_diff_2_neg)
sigma_neg_squared = sigma_neg_squared/(2*N_neg)


# plot decision boundary

import matplotlib.pyplot
from numpy import arange
from numpy import meshgrid

# Create a grid of points
xrange = arange(min(df[0]), max(df[0]), 0.01)
yrange = arange(min(df[1]), max(df[1]), 0.01)
X, Y = meshgrid(xrange,yrange)

# Implicit Function for Decision Boundary
F = (1/sigma_pos_squared)*np.exp(-1/(2*sigma_pos_squared)*((X - mu_pos[0])**2 + (Y - mu_pos[1])**2)) # implicit function LHS
G = (1/sigma_neg_squared)*np.exp(-1/(2*sigma_neg_squared)*((X - mu_neg[0])**2 + (Y - mu_neg[1])**2)) # implicit function RHS

from matplotlib.colors import ListedColormap

# Plot Data Points
plt.scatter(first_part[0], first_part[1], color = 'red', label = 'Positive class')
plt.scatter(second_part[0], second_part[1], color = 'blue', label = 'Negative class')

# Plot Contour
CS = plt.contour(X, Y, (F - G), levels = [0], alpha = 0.8, cmap = ListedColormap('black'))
labels = ['Decision Boundary']

for i in range(len(labels)):
    CS.collections[i].set_label(labels[i])

# Label Plot
plt.title('Generative Classification with Gaussian Class-Conditionals (different covariances)')
plt.xlabel('X1 (feature 1)')
plt.ylabel('X2 (feature 2)')
plt.legend()
plt.show()

# 2. Generative Classification with Gaussian Class-Conditionals (same covariances)

# find (sigma)^2

X_diff_1_pos = np.array(first_part[0] - mu_pos[0]) # first coordinate
X_diff_2_pos = np.array(first_part[1] - mu_pos[1]) # second coordinate

X_diff_1_pos = X_diff_1_pos**2
X_diff_2_pos = X_diff_2_pos**2

sigma_pos_squared = sum(X_diff_1_pos) + sum(X_diff_2_pos)

X_diff_1_neg = np.array(second_part[0] - mu_neg[0]) # first coordinate
X_diff_2_neg = np.array(second_part[1] - mu_neg[1]) # second coordinate

X_diff_1_neg = X_diff_1_neg**2
X_diff_2_neg = X_diff_2_neg**2

sigma_neg_squared = sum(X_diff_1_neg) + sum(X_diff_2_neg)

sigma_squared = (sigma_pos_squared + sigma_neg_squared)/(2*(N_pos + N_neg))


# plot decision boundary

import matplotlib.pyplot
from numpy import arange
from numpy import meshgrid

# Create a grid of points
xrange = arange(min(df[0]), max(df[0]), 0.01)
yrange = arange(min(df[1]), max(df[1]), 0.01)
X, Y = meshgrid(xrange,yrange)

# Implicit Function for Decision Boundary
F = (1/sigma_squared)*np.exp(-1/(2*sigma_squared)*((X - mu_pos[0])**2 + (Y - mu_pos[1])**2)) # implicit function LHS
G = (1/sigma_squared)*np.exp(-1/(2*sigma_squared)*((X - mu_neg[0])**2 + (Y - mu_neg[1])**2)) # implicit function RHS

from matplotlib.colors import ListedColormap

# Plot Data Points
plt.scatter(first_part[0], first_part[1], color = 'red', label = 'Positive class')
plt.scatter(second_part[0], second_part[1], color = 'blue', label = 'Negative class')

# Plot Contour
CS = plt.contour(X, Y, (F - G), levels = [0], alpha = 0.8, cmap = ListedColormap('black'))
labels = ['Decision Boundary']

for i in range(len(labels)):
    CS.collections[i].set_label(labels[i])

# Label Plot    
plt.title('Generative Classification with Gaussian Class-Conditionals (same covariances)')
plt.xlabel('X1 (feature 1)')
plt.ylabel('X2 (feature 2)')
plt.legend()
plt.show()

# 3. Linear SVM
from sklearn.svm import SVC
classifier = SVC(C = 1, kernel = 'linear', random_state = 0)
classifier.fit(df.iloc[:, [0, 1]].values, df.iloc[:, [2]].values)

# plot decision boundary
from matplotlib.colors import ListedColormap
X_set, y_set = df.iloc[:, [0, 1]].values, df.iloc[:, [2]].values

# Create a grid of points
xrange = arange(min(df[0]), max(df[0]), 0.01)
yrange = arange(min(df[1]), max(df[1]), 0.01)
X, Y = meshgrid(xrange,yrange)

# Plot Data Points
plt.scatter(first_part[0], first_part[1], color = 'red', label = 'Positive class')
plt.scatter(second_part[0], second_part[1], color = 'blue', label = 'Negative class')

# Plot Contour
plt.contourf(X, Y, classifier.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape), alpha = 0.3)

# Label Plot
plt.title('Linear SVM')
plt.xlabel('X1 (feature 1)')
plt.ylabel('X2 (feature 2)')
plt.legend()
plt.show()