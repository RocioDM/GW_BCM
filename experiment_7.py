#This notebook recovers the weights in the analysis problem of GW - barycenters and tests accuracy on PointCloud MNIST dataset.

import numpy as np  # linear algebra
import pandas as pd  # data processing
import scipy as sp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import time
import kagglehub
import ot



## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils



## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()



## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
digit = 3  # Pick a digit from 0 to 9
n_temp = 3  # Number of templates
ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices

for s in range(n_temp):
    # Select a random index corresponding to the chosen digit
    ind = digit_indices[digit][np.random.randint(len(digit_indices[digit]))]
    ind_temp_list.append(ind)

    # Extract the probability measure from the third column of Data (p_s)
    p_s = Data[ind, :, 2]

    # Find valid indices where p_s is not -1 (avoiding missing or padded values)
    valid_indices = np.where(p_s != -1)[0]

    # Filter out invalid entries from p_s
    p_s = p_s[valid_indices]

    # Normalize p_s to make it a valid probability distribution
    p_s = p_s / float(p_s.sum())
    measure_temp_list.append(p_s)

    # Extract spatial coordinates (first two columns of Data) for valid points
    C_s = Data[ind, valid_indices, :2]

    # Center the points by subtracting the mean
    # and Normalize coordinates to fit within the unit square [0,1]²
    C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

    # Compute the pairwise Euclidean distance matrix for C_s
    dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
    matrix_temp_list.append(dist_matrix_s)



## GENERATING RANDOM VECTOR OF WEIGHTS AND SYNTHESIZING A BARYCENTER USING POT ####################
# Random vector of weights
lambdas_list = np.random.rand(n_temp)
lambdas_list = lambdas_list / lambdas_list.sum()

# Synthesize a Barycenter using POT
M = 100  # Dimension of output barycentric matrix is MxM.

# b = np.ones(M) / M   # Uniform target probability vector
b = np.random.rand(M)
b = b / b.sum()  # Random target probability vector

B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 lambdas_list)  # Synthesize barycenter matrix
B = (B + B.T) / 2  # enforce symmetry of synthesized barycenter



## RECOVER VECTOR OF WEIGHTS AND RECONSTRUCTED BARYCENTER B_RECON FROM SYNTHESIZED BARYCENTER USING
# utils.get_lambdas FUNCTION AND COMPUTING ERROR ##################################################
B_recon, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
B_recon = (B_recon + B_recon.T) /2 # Enforce Symmetry of the Reconstructed Barycenter matrix
print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))



## COMPARING POT-SYNTHESIZED BARYCENTER B VS RECONSTRUCTED BARYCENTER B-RECON BY COMPUTING THE
# GW-DISTANCE BETWEEN THEM ########################################################################
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')



## PREPROCESS B AND B_RECON USING MDS TO VISUALIZE LATER ##########################################
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)  # Create an MDS instance

# Apply MDS embedding to Barycenter B and Reconstructed Barycenter B_recon,
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

# Center and fit points to be in the [0,1]x[0,1] square for visualization
points_B = utils.normalize_2Dpointcloud_coordinates(points_B)
points_B_recon = utils.normalize_2Dpointcloud_coordinates(points_B_recon)



## PLOT TEMPLATES #################################################################################
fig, axes = plt.subplots(1, n_temp, figsize=(12, 6))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind, :, 2]
    X = Data[ind, a != -1, :2]
    X = utils.normalize_2Dpointcloud_coordinates(X)
    a = a[a != -1]
    a = a / float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
    axes[i].set_title(f'Template #{i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    # axes[i].axis('off')

plt.show()



## PLOT B AND B_RECON #############################################################################
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(points_B[:, 0], points_B[:, 1], s=b * 250)
axes[0].set_title('Input Barycenter')
axes[0].set_xticks([])  # Remove x-axis ticks
axes[0].set_yticks([])  # Remove y-axis ticks

axes[1].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 250)
axes[1].set_title('Reconstructed Barycenter')
axes[1].set_xticks([])  # Remove x-axis ticks
axes[1].set_yticks([])  # Remove y-axis ticks

plt.show()