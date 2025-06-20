## Analysis Problem of GW barycenter for Point Clouds Corruption (Occlusion)
## This notebook recovers the weights in the analysis problem of
## GW - barycenters from an occluded sample and occluded templates
## and recover the sample using the recovered weights and non-occluded templates.
## The input sample is taken in the Barycenter space of the non-occluded templates and then
## corrupted under occlusion (circular mask).

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

import ot   # POT: Python Optimal Transport library


## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## USER DEFINED FUNCTIONS #########################################################################
def occlusion_circular(X, a):
    '''
    Applies an occlusion mask using a circular region to a set of 2D points and their corresponding weights.

    Input:
    :param X: (numpy array of shape (N,2)) A set of N 2D points (point coordinates).
    :param a: (numpy array of shape (N,)) A set of N weights corresponding to each point.

    Output:
    :return X_occluded: (numpy array) The points that remain after occlusion.
    :return a_occluded: (numpy array) The corresponding weights, renormalized to sum to 1.
    '''
    # Define occlusion circle
    circle_center = np.array([0.5, 0.5])  # Center of the occlusion
    radius = 0.2  # Radius of the occlusion circle

    # Compute distance from each point to the circle center
    distances = np.linalg.norm(X - circle_center, axis=1)

    # Mask points outside the circle
    mask = distances > radius

    # Apply mask to X and a
    X_occluded = X[mask]  # Keep only points outside the occlusion circle
    a_occluded = a[mask]  # Remove mass inside the circle

    # Re-normalize `a` so it sums to 1
    a_occluded /= a_occluded.sum()

    return X_occluded, a_occluded


## DATASET LOADING ################################################################################
Data, label, digit_indices = utils.load_pointcloudmnist2d()

## TEST THE OCCLUSION FUNCTION IN ONE SAMPLE ######################################################
print(f'First, you will visualize a simulated occlusion of a portion of a point cloud')
# Select a random sample
u = np.random.randint(0, 100)
a = Data[u, :, 2]  # Original mass values
X = Data[u, a != -1, :2]  # Extract valid points

# Normalize X: center and scaled to fit within the unit square [0,1] x [0,1]
X = utils.normalize_2Dpointcloud_coordinates(X)

# Filter `a` (only keep valid entries)
a = a[a != -1]
a = a / float(a.sum())  # Normalize to sum to 1

# Apply circular occlusion
X_occluded, a_occluded = occlusion_circular(X, a)

# Plot before and after occlusion
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X[:, 0], X[:, 1], s=a * 500, color='blue')
ax[0].set_title("Original Point Cloud")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)
ax[0].set_xticks([])  # Remove x-axis ticks
ax[0].set_yticks([])  # Remove y-axis ticks

ax[1].scatter(X_occluded[:, 0], X_occluded[:, 1], s=a_occluded * 500, color='blue')
ax[1].set_title("Occluded Point Cloud")
ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, 1)
ax[1].set_xticks([])  # Remove x-axis ticks
ax[1].set_yticks([])  # Remove y-axis ticks

# Draw occlusion circle
circle = plt.Circle((0.5, 0.5), 0.2, edgecolor='red', facecolor='none', linestyle='--', lw=2)
ax[0].add_patch(circle)
ax[1].add_patch(plt.Circle((0.5, 0.5), 0.2, edgecolor='red', facecolor='none', linestyle='--', lw=2))

plt.show()

## GET RANDOM TEMPLATES AND THEIR OCCLUSIONS FROM DATASET #########################################

print('Selecting random templates and simulating their occlusions')

# Templates are of the form (matrix, measure)
digit = 5  # Pick a digit from 0 to 9
n_temp = 3  # Number of templates

ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices
measure_occ_list = []  # list of template measures occluded
matrix_occ_list = []  # list of template dissimilarity matrices occluded

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

    # Apply occlusion to the spatial coordinates and probability measure
    C_occluded, p_occluded = occlusion_circular(C_s, p_s)
    measure_occ_list.append(p_occluded)

    # Compute the distance matrix for the occluded points
    dist_matrix_occ = sp.spatial.distance.cdist(C_occluded, C_occluded)
    matrix_occ_list.append(dist_matrix_occ)

print('Synthesizing a GW-barycenter using POT and perturbing it by occlusion')

## GENERATE A RANDOM VECTOR OF WEIGHTS, SYNTHESIZING A BARYCENTER USING POT AND ITS OCCLUSION #####
# Random vector of weights
# lambdas_list = np.random.rand(n_temp)
# lambdas_list = lambdas_list / lambdas_list.sum()
lambdas_list = np.random.dirichlet(np.ones(n_temp), size=1)[0]

# Create an MDS instance
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

# Synthesize a Barycenter using POT
#M = 100
M = len(C_s) # Dimension of output barycentric matrix is MxM.

b = np.ones(M) / M   # Uniform target probability vector
# b = np.random.rand(M)
# b = b / b.sum()  # Random target probability vector

B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 lambdas_list)  # Synthesize barycenter matrix
B = (B + B.T) / 2  # enforce symmetry

# Center and fit points to be in the [0,1]x[0,1] square for later visualization
points_B = mds.fit_transform(B, init=C_s)
points_B = utils.normalize_2Dpointcloud_coordinates(points_B)

## Occluding the barycenter and compute distance matrix
B1, b1 = occlusion_circular(points_B, b)
dist_matrix_occ = sp.spatial.distance.cdist(B1, B1)









# ###################################################################################################
#
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
# # Apply occlusion to the spatial coordinates and probability measure
# B1, b1 = occlusion_circular(points_B, b)
# dist_matrix_occ = sp.spatial.distance.cdist(B1, B1)
#
# ###################################################################################################



## RECOVER VECTOR OF WEIGHTS FROM OCCLUDED SYNTHESIZED BARYCENTER USING utils.get_lambdas FUNCTION,
#  RECONSTRUCTED BARYCENTER B_RECON USING POT AND NON-OCCLUDED TEMPLATES, AND COMPUTE ERRORS ######

print('Estimating the vector lambda from the perturbed input with perturbed templates')

_, lambdas = utils.get_lambdas(matrix_occ_list, measure_occ_list, dist_matrix_occ, b1)



print('Reconstruction of the input from the estimated lambda vector and using unperturbed templates (using POT for synthesis)')
M = len(b)
B_recon = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas)
B_recon = (B_recon + B_recon.T) / 2  # sym


## Compare Original target vs reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist_analysis = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist_analysis:.4f}')

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

# Draw occlusion box
for a in axes:
    a.add_patch(
        plt.Circle((0.5, 0.5), 0.2, edgecolor='red', facecolor='none', linestyle='--', lw=2))
plt.show()


## Random reconstruction
U = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 np.random.dirichlet(np.ones(n_temp), size=1)[0])

## Compare Original target vs random reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, U, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']

print(f'GW(Target,Random Reconstruction): {gw_dist:.4f}')

U = mds.fit_transform(U)
U = utils.normalize_2Dpointcloud_coordinates(U)


## PLOT Synthesized Barycenter (B) AND Reconstructed Barycenter (B_RECON) #########################
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].scatter(points_B[:, 0], points_B[:, 1], s=b * 500)
axes[0].set_title('Input Barycenter')
axes[0].set_xticks([])  # Remove x-axis ticks
axes[0].set_yticks([])  # Remove y-axis ticks

axes[1].scatter(B1[:, 0], B1[:, 1], s=b1 * 500)
axes[1].set_title('Occluded Barycenter')
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




n_iter = 100  # Number of random reconstructions to perform
gw_distances = []

for i in range(n_iter):
    # Random reconstruction
    U = ot.gromov.gromov_barycenters(
        M, matrix_temp_list, measure_temp_list, b,
        np.random.dirichlet(np.ones(n_temp), size=1)[0]
    )

    # Compare Original target vs random reconstruction
    gromov_distance = ot.gromov.gromov_wasserstein(B, U, b, b, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    gw_distances.append(gw_dist)

    #print(f'[{i+1}] GW(Target, Random Reconstruction): {gw_dist:.4f}')



# a = gw_dist_analysis
#
# plt.figure(figsize=(8, 5))
# plt.plot(gw_distances, marker='o', label='GW(Target, Random Reconstruction)')
# plt.axhline(y=a, color='r', linestyle='--', label=f'GW(Target, Proposed Reconstruction) = {a}')
# plt.xlabel('Iteration')
# plt.ylabel('GW Distance')
# #plt.title('GW Distances')
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.show()


a = gw_dist_analysis

plt.figure(figsize=(8, 5))
plt.plot(gw_distances, 'o', color='blue', label='GW(Target, Random Reconstruction)')  # Only blue dots
plt.axhline(y=a, color='r', linestyle='--', label=f'GW(Target, Proposed Reconstruction) = {a:.4f}')
plt.xlabel('Iteration')
plt.ylabel('GW Distance')
# plt.title('GW Distances')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()