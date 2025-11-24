## 3D Data
## This notebook recovers the weights in the analysis problem of GW-barycenters
## and tests accuracy on a 3D PointCloud dataset.
## We use the function "get_lambdas" from "utils" (Fix-Point approach)
## and also the functions "blow_up" and "get_lambdas_blowup" from "utils" (Blow-up approach)
## GW-Barycenters are synthesized using the function "ot.gromov.gromov_barycenters" from the POT Library

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
from sklearn.manifold import MDS

import ot   # POT: Python Optimal Transport library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


## GET TEMPLATES
print('Getting 3D point cloud templates from the same class')

## GET TEMPLATES
# List of different airplane sample files
airplane_files = [
    'airplane_0236.off',
    'airplane_0435.off',
    'airplane_0215.off'
]

# number of templates
n_temp = len(airplane_files)


# Bounds for sample points from the mesh surface
l_bound = 400
u_bound = 401


# Store the sampled points for each airplane
sampled_data = []
# list of dissimilarity matrices
matrix_temp_list = []
# list of measures
measure_temp_list = []

# Loop through each airplane file and sample points
for airplane_file in airplane_files:
    # Construct the full path to the .off file
    sample_file_path = os.path.join(dataset_path, 'ModelNet40', 'airplane', 'train', airplane_file)

    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(sample_file_path)

    # Random number of samples
    num_points_to_sample = np.random.randint(l_bound, u_bound)
    # Sample points from the mesh surface
    sampled_points = mesh.sample(num_points_to_sample)

    # Normalize the points to fit within [0, 1]^3
    min_vals = sampled_points.min(axis=0)
    max_vals = sampled_points.max(axis=0)
    normalized_points = (sampled_points - min_vals) / (max_vals - min_vals)

    # Append the normalized points to the list
    sampled_data.append(normalized_points)

    # Dissimilarity matrices
    dist_matrix = sp.spatial.distance.cdist(normalized_points, normalized_points)
    matrix_temp_list.append(dist_matrix)

    # Measure
    p_s = np.ones(num_points_to_sample) / num_points_to_sample
    measure_temp_list.append(p_s)


## Get vector of weights ##########################################################################
#lambdas_list =  np.random.dirichlet(np.ones(n_temp), size=1)[0] # generates random samples from a Dirichlet distribution, which is a common way to generate probability distributions over a simplex.
lambdas_list = np.array([1/3,1/3,1/3])  # Uniform


## Synthesize a GW-Barycenter using POT ###########################################################
print('Synthesizing a GW-Barycenter using the POT Library')
M = 400 # Dimension of output barycentric matrix is MxM.

b = np.ones(M) / M   # Uniform target probability vector
# b = np.random.rand(M)
# b = b/b.sum()   # Random target probability vector

B =  ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas_list, max_iter=5000, tol=1e-16)  # Synthesize barycenter matrix

# Create an MDS instance for visualization
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
# Fit and transform the distance matrix
points_B = mds.fit_transform(B)


## Recover the vector of weights 'lambdas' by only knowing the Barycenter #########################
print('Solving the GW-analysis problem with the two proposed methods')

## From the fix-point approach:
B_recon, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
## Fit and transform the distance matrix through MDS
points_B_recon = mds.fit_transform(B_recon)


## From the blow-up approach
B_blow_up, b_blow_up, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
print('Size of the blow-up: ', B_blow_up.shape[0])
points_B_blowup = mds.fit_transform(B_blow_up,B)
B_recon_blow_up, lambdas_recon_blow_up = utils.get_lambdas_blowup(temp_blow_up, B_blow_up, b_blow_up)
## Fit and transform the distance matrix through MDS
points_B_recon_blow_up = mds.fit_transform(B_recon_blow_up)



## Print accuracy of the lambdas-recovery #########################################################

## Fix point method:
print('Accuracy obtained from the fixed point approach')

## Print lambda-vectors: original, after analysis, error
print('Original lambda-vector = ', lambdas_list)
print('Recovered lambda-vector = ', lambdas)
print('Error = ', np.linalg.norm(lambdas_list - lambdas, 1))

## Compare Original target vs reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')


## Blow-up method:
print('Accuracy obtained from the blow-up approach')

## Print lambda-vectors: original, after analysis, error
print('Original lambda-vector = ', lambdas_list)
print('Recovered lambda-vector = ', lambdas_recon_blow_up)
print('Error = ', np.linalg.norm(lambdas_list - lambdas_recon_blow_up, 1))

## Compare Original target vs. its blow-up version
gromov_distance = ot.gromov.gromov_wasserstein(B, B_blow_up, b, b_blow_up, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Target Blow-up): {gw_dist}')

## Compare target blow-up vs. reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B_blow_up, B_recon_blow_up, b_blow_up, b_blow_up, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target Blow-up, Reconstructed Target): {gw_dist}')

## Compare Original target vs. reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon_blow_up, b, b_blow_up, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Original Target,Reconstructed Target): {gw_dist}')



## PLOT ###########################################################################################

# Plot templates and their blowups
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})

# Set the main title for the entire figure
fig.suptitle("Templates and Blow-Ups", fontsize=18)

# Plot of original sampled templates
for i, sampled_points in enumerate(sampled_data[:3]):
    ax = axes[0, i]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)
    ax.set_title(f"Airplane {i + 1}")
    ax.set_axis_off()
    ax.grid(False)

# Plot of blow-up templates
for i in range(n_temp):
    sampled_points = mds.fit_transform(temp_blow_up[i], sampled_data[i:3])
    ax = axes[1, i]
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)
    ax.set_title(f"Airplane Blow-Up {i + 1}")
    ax.set_axis_off()
    ax.grid(False)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()



## PLOT SYNTHETIC BARYCENTER AND THE TWO RECONSTRUCTIONS
fig, axes = plt.subplots(2, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

# subplot: Original Synthesized GW-Barycenter
axes[0,0].scatter(points_B[:, 0], points_B[:, 1], points_B[:, 2], s=1)
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
axes[0,0].set_zlabel('Z')
axes[0,0].set_title('Synthesized GW Barycenter')
axes[0,0].set_axis_off()
axes[0,0].grid(False)

# subplot: Blow-up
axes[0,1].scatter(points_B_blowup[:, 0], points_B_blowup[:, 1], points_B_blowup[:, 2], s=1)
axes[0,1].set_xlabel('X')
axes[0,1].set_ylabel('Y')
axes[0,1].set_zlabel('Z')
axes[0,1].set_title('Blow-Up GW Barycenter')
axes[0,1].set_axis_off()
axes[0,1].grid(False)

# subplot: Reconstruction via fixed-point approach
axes[1,0].scatter(points_B_recon[:, 0], points_B_recon[:, 1], points_B_recon[:, 2], s=1)
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Y')
axes[1,0].set_zlabel('Z')
axes[1,0].set_title('Reconstruction (Fixed-Point Approach)')
axes[1,0].set_axis_off()
axes[1,0].grid(False)

# subplot: Reconstruction via blow-up approach
axes[1,1].scatter(points_B_recon_blow_up[:, 0], points_B_recon_blow_up[:, 1], points_B_recon_blow_up[:, 2], s=1)
axes[1,1].set_xlabel('X')
axes[1,1].set_ylabel('Y')
axes[1,1].set_zlabel('Z')
axes[1,1].set_title('Reconstruction (Blow-Up Approach)')
axes[1,1].set_axis_off()
axes[1,1].grid(False)

# Adjust layout
plt.tight_layout()

# Show the figure with both subplots
plt.show()