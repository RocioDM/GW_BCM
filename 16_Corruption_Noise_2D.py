## Analysis Problem of GW barycenter for Point Clouds Corruption (Additive Noise)
## This notebook recovers the weights in the analysis problem of
## GW - barycenters from noisy templates
## and recover the sample using the recovered weights and clean templates.
## The input sample is taken in the Barycenter space of the clean templates and then
## corrupted under noise

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

import ot   # POT: Python Optimal Transport library


## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## USER DEFINED FUNCTIONS #########################################################################
def add_white_noise(X, std_dev=0.05):
    '''
    Adds Gaussian white noise to a set of 2D points and returns the perturbed point cloud and same weights.

    Input:
    :param X: (numpy array of shape (N,2)) A set of N 2D points.
    :param std_dev: (float) Standard deviation of the Gaussian noise.

    Output:
    :return X_noisy: (numpy array) The noisy points.
    :return a: (numpy array) The unchanged weights.
    '''
    noise = np.random.normal(loc=0.0, scale=std_dev, size=X.shape)
    X_noisy = X + noise
    X_noisy = utils.normalize_2Dpointcloud_coordinates(X_noisy)
    return X_noisy


## DATASET LOADING ################################################################################
Data, label, digit_indices = utils.load_pointcloudmnist2d()

## TEST THE NOISE FUNCTION IN ONE SAMPLE ######################################################
print(f'First, you will visualize a simulated noisy point cloud')
# Select a random sample
u = np.random.randint(0, 100)
a = Data[u, :, 2]  # Original mass values
X = Data[u, a != -1, :2]  # Extract valid points

# Normalize X: center and scaled to fit within the unit square [0,1] x [0,1]
X = utils.normalize_2Dpointcloud_coordinates(X)

# Filter `a` (only keep valid entries)
a = a[a != -1]
a = a / float(a.sum())  # Normalize to sum to 1

# Apply additive noise
X_noisy = add_white_noise(X)

# Plot before and after additive noise
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X[:, 0], X[:, 1], s=a * 500, color='blue')
ax[0].set_title("Original Point Cloud")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)

l = X_noisy.shape[0]
a = np.ones(l)/l

ax[1].scatter(X_noisy[:, 0], X_noisy[:, 1], s=a * 500 , color='blue')
ax[1].set_title("Noisy Point Cloud")
ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, 1)

plt.show()



## GET RANDOM TEMPLATES AND THEIR NOISE CORRUPTIONS FROM DATASET ##################################

print('Selecting random templates and simulating their corruption under noise')

# Templates are of the form (matrix, measure)
digit = 3  # Pick a digit from 0 to 9
n_temp = 5  # Number of templates

ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices
matrix_noise_list = []  # list of template dissimilarity matrices with additive noise

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

    # Apply additive noise to the spatial coordinates and probability measure
    C_noisy = add_white_noise(C_s)

    # Compute the distance matrix for the noisy points
    dist_matrix_noise = sp.spatial.distance.cdist(C_noisy, C_noisy)
    matrix_noise_list.append(dist_matrix_noise)

print('Synthesizing a GW-barycenter using POT and perturbing it by additive noise')


## GENERATE A RANDOM VECTOR OF WEIGHTS, SYNTHESIZING A BARYCENTER USING POT AND ITS NOISY VERSIONS
# Random vector of weights
# lambdas_list = np.random.rand(n_temp)
# lambdas_list = lambdas_list / lambdas_list.sum()
lambdas_list = np.random.dirichlet(np.ones(n_temp), size=1)[0]

# Create an MDS instance
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

# Synthesize a Barycenter using POT
#M = 150
M = len(C_s) # Dimension of output barycentric matrix is MxM.

b = np.ones(M) / M   # Uniform target probability vector
# b = np.random.rand(M)
# b = b / b.sum()  # Random target probability vector

B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 lambdas_list)  # Synthesize barycenter matrix
B = (B + B.T) / 2  # enforce symmetry (optional)

# Center and fit points to be in the [0,1]x[0,1] square for later visualization
points_B = mds.fit_transform(B, init=C_s)
points_B = utils.normalize_2Dpointcloud_coordinates(points_B)

## Noisy barycenter and compute distance matrix
B1 = add_white_noise(points_B)
dist_matrix_noise = sp.spatial.distance.cdist(B1, B1)

l = B1.shape[0]
b1 = np.ones(l)/l





# # ###################################################################################################
# #
# # Select a random index corresponding to the chosen digit
# ind = digit_indices[digit][np.random.randint(len(digit_indices[digit]))]
# # Extract the probability measure from the third column of Data (p_s)
# b = Data[ind, :, 2]
# valid_indices = np.where(b != -1)[0]
# b = b[valid_indices]
# b = b / float(b.sum())
#
# B = Data[ind, valid_indices, :2]
# points_B = utils.normalize_2Dpointcloud_coordinates(B)
# B = sp.spatial.distance.cdist(B, B)
#
# # Apply additive noise to the spatial coordinates and probability measure
# B1 = add_white_noise(points_B)
# dist_matrix_noise = sp.spatial.distance.cdist(B1, B1)
# l = dist_matrix_noise.shape[0]
# b1 = np.ones(l)/l
# #
# # ###################################################################################################





## RECOVER VECTOR OF WEIGHTS FROM NOISY SYNTHESIZED BARYCENTER USING utils.get_lambdas FUNCTION,
#  RECONSTRUCTED BARYCENTER B_RECON USING POT AND NON-NOISY TEMPLATES, AND COMPUTE ERRORS ######

print('Estimating the vector lambda from the perturbed input with perturbed templates')

_, lambdas = utils.get_lambdas(matrix_noise_list, measure_temp_list, dist_matrix_noise, b1)



print('Reconstruction of the input from the estimated lambda vector and using unperturbed templates (using POT for synthesis)')
M = len(b)
B_recon = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas)
B_recon = (B_recon + B_recon.T) / 2  # sym


## Compare Original target vs reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')

## Fit and transform the distance matrix of B_recon
points_B_recon = mds.fit_transform(B_recon, init=points_B)
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
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 500)
    axes[i].set_title(f'Template {i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)
    # axes[i].axis('off')

plt.show()


U = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 np.random.dirichlet(np.ones(n_temp), size=1)[0])
U = mds.fit_transform(U)
U = utils.normalize_2Dpointcloud_coordinates(U)


## PLOT Synthesized Barycenter (B) AND Reconstructed Barycenter (B_RECON) #########################
fig, axes = plt.subplots(1, 4, figsize=(12, 4))

axes[0].scatter(points_B[:, 0], points_B[:, 1], s=b * 500)
axes[0].set_title('Input Barycenter')
axes[0].set_xticks([])  # Remove x-axis ticks
axes[0].set_yticks([])  # Remove y-axis ticks

axes[1].scatter(B1[:, 0], B1[:, 1], s=b1 * 500)
axes[1].set_title('Noisy Barycenter')
axes[1].set_xticks([])  # Remove x-axis ticks
axes[1].set_yticks([])  # Remove y-axis ticks

axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 500)
axes[2].set_title('Reconstructed Barycenter')
axes[2].set_xticks([])  # Remove x-axis ticks
axes[2].set_yticks([])  # Remove y-axis ticks

axes[3].scatter(U[:, 0], U[:, 1], s=b * 350)
axes[3].set_title('Random Reconstruction')
axes[3].set_xticks([])  # Remove x-axis ticks
axes[3].set_yticks([])  # Remove y-axis ticks

plt.show()
