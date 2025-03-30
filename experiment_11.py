## Classification experiment: "MAXIMUM COORDINATE"
## We randomly select templates from each class of the dataset ('k' samples per class).
## Given a random point from the dataset, we compute its GW-Barycentric coordinates.
## We classify the input into the class corresponding to the template-label with the
## highest coordinate in the vector of GW-Barycentric coordinates ('lambdas').
## Finally, we compute the accuracy of the method.

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils



## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()



##Get templates: Pick one random sample point cloud for each digit.
## Templates (matrix, measure)

n_classes = 10  # Since we are working with digits 0-9

k = 3   # Number of templates of each class

## test: see which one of the 10 classes does the mass most concentrate on

n_temp = k * 10   # Number of templates in total

ind_temp_list = []   #list of template indices from dataset
measure_temp_list = []   #list of template measures
matrix_temp_list = []   #list of template dissimilarity matrices

for digit in range(n_classes):
    for s in range(k):
        ind = digit_indices[digit][np.random.randint(len(digit_indices[digit]))]
        ind_temp_list.append(ind)

        p_s = Data[ind,:,2]
        valid_indices = np.where(p_s != -1)[0]
        p_s = p_s[valid_indices]
        p_s = p_s / float(p_s.sum())
        measure_temp_list.append(p_s)

        C_s = Data[ind, valid_indices, :2]
        C_s = C_s-C_s.mean(0)[np.newaxis,:]
        dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
        matrix_temp_list.append(dist_matrix_s)


## Plot templates
fig, axes = plt.subplots(1, n_temp, figsize=(12, 6))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind,:,2]
    X = Data[ind,a!=-1,:2]
    X = X-X.mean(0)[np.newaxis,:]
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    a = a[a!=-1]
    a = a/float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a*250)
    #axes[i].set_title(f'Digit #{i}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    #axes[i].axis('off')

# Add figure title
fig.suptitle("Templates", fontsize=16)
plt.show()


##TEST
print('Testing  in one sample of the data set:')
print(f'As templates we are using {k} random samples of digit point-clouds from 0 to 9.')
# Select a random sample
u = np.random.randint(0, 100)
a = Data[u, :, 2]  # Original mass values
X = Data[u, a != -1, :2]  # Extract valid points

# Normalize X
X = X - X.mean(0)[np.newaxis, :]
dist_matrix_input = sp.spatial.distance.cdist(X, X)

# Filter `a` (only keep valid entries)
a = a[a != -1]
a = a / float(a.sum())  # Normalize to sum to 1



print('label of input = ', label[u])
_, lambdas = utils.get_lambdas(matrix_temp_list,measure_temp_list,dist_matrix_input,a)
print('lambda vector = ', lambdas)
computed_label = np.where(lambdas == lambdas.max())[0][0]
computed_label = computed_label // k
print('max lambda class = ',computed_label)
if label[u] == computed_label:
    print('This random input is correctly classified.')
else:
    print('This random input is NOT correctly classified.')



##COMPUTE ACCURACY
positive_case = 0

n_ite = 1000 # max = len(label)

print(f'Now we will compute the accuracy of the classification for {n_ite} points of the data set:')

for i in range(n_ite):
  #Input test
  a = Data[i, :, 2]  # Original mass values
  X = Data[i, a != -1, :2]  # Extract valid points
  # Normalize X
  X = X - X.mean(0)[np.newaxis, :]
  dist_matrix_input = sp.spatial.distance.cdist(X, X)
  # Filter `a` (only keep valid entries)
  a = a[a != -1]
  a = a / float(a.sum())  # Normalize to sum to 1

  _, lambdas = utils.get_lambdas(matrix_temp_list,measure_temp_list,dist_matrix_input,a)

  computed_lambda = np.where(lambdas == lambdas.max())[0][0]
  computed_lambda = computed_lambda // k

  if label[i] == computed_lambda:
    positive_case +=1

  # Print progress every 50 iterations
  if i % 50 == 0:
    print(f"Processed {i} samples...")
    if i != 0:
     print(f'So far, the accuracy of the method is {positive_case/i}')


print('Accuracy = ', positive_case/n_ite)