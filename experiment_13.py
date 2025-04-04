import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
from sklearn.manifold import MDS

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils

# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


## EXAMPLE AND VISUALIZATION ######################################################################
# Example category and sample file name
category = 'airplane'  # Replace with the desired category
sample_file = 'airplane_0236.off'  # Replace with the actual .off file

# Construct the full path to the .off file
sample_file_path = os.path.join(dataset_path, 'ModelNet40', category, 'train', sample_file)

# Load the mesh using trimesh
mesh = trimesh.load_mesh(sample_file_path)

# Sample points from the mesh surface (e.g., 1000 points)
num_points_to_sample = 1000
sampled_points = mesh.sample(num_points_to_sample)

# Plot the sampled points in 3D using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_axis_off()
ax.grid(False)

plt.show()


## GET TEMPLATES
#number of templates
n_temp = 3

# List of 5 different airplane sample files
airplane_files = [
    'airplane_0236.off',
    'airplane_0435.off',
    'airplane_0215.off',
    #'airplane_0162.off',
    #'airplane_0303.off',
]  # Replace with the actual files in your dataset

# Number of points to sample from each airplane
num_points_to_sample = 1000

# Store the sampled points for each airplane
sampled_airplanes = []
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

    # Sample points from the mesh surface
    sampled_points = mesh.sample(num_points_to_sample)

    # Normalize the points to fit within [0, 1]^3
    min_vals = sampled_points.min(axis=0)
    max_vals = sampled_points.max(axis=0)
    normalized_points = (sampled_points - min_vals) / (max_vals - min_vals)

    # Append the normalized points to the list
    sampled_airplanes.append(normalized_points)

    # Dissimilarity matrices
    dist_matrix = sp.spatial.distance.cdist(normalized_points, normalized_points)
    matrix_temp_list.append(dist_matrix)

    # Measure
    p_s = np.ones(num_points_to_sample) / num_points_to_sample
    measure_temp_list.append(p_s)



# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

# Set the main title for the entire figure
fig.suptitle("Templates", fontsize=16)

# Plot each normalized airplane in a separate subplot
for i, (ax, sampled_points) in enumerate(zip(axes, sampled_airplanes[:3])):  # Use first 3 airplanes
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Airplane {i + 1}")

    ax.set_axis_off()
    ax.grid(False)

plt.tight_layout()

plt.show()

## Get vector of weights
lambdas_list = np.random.rand(n_temp)
lambdas_list = lambdas_list/lambdas_list.sum()


#Synthesize a Barycenter using POT
M = 250 # Dimension of output barycentric matrix is MxM.

b = np.ones(M) / M   # Uniform target probability vector
# b = np.random.rand(M)
# b = b/b.sum()   # Random target probability vector

B =  ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas_list)  # Synthesize barycenter matrix


## Recover the vector of weights 'lambdas' by only knowing the Barycenter
B_recon, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))
print('lambdas = ', lambdas)


## VISUALIZATION
# Create an MDS instance
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)

## Fit and transform the distance matrix
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

## PLOT
# Create a single figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

# First subplot: Original sampled points
axes[0].scatter(points_B[:, 0], points_B[:, 1], points_B[:, 2], s=1)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_zlabel('Z')
axes[0].set_title('Original Points')
axes[0].set_axis_off()
axes[0].grid(False)

# Second subplot: Reconstructed sampled points
axes[1].scatter(points_B_recon[:, 0], points_B_recon[:, 1], points_B_recon[:, 2], s=1)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_zlabel('Z')
axes[1].set_title('Reconstructed Points')
axes[1].set_axis_off()
axes[1].grid(False)

# Adjust layout
plt.tight_layout()

# Show the figure with both subplots
plt.show()