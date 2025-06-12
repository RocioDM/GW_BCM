## 3D Data: Occlusion experiment

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


## USER DEFINED FUNCTIONS #################################################################
def occlusion_spherical(X, a, radius = 0.2):
    '''
    Applies an occlusion mask using a spherical region to a set of 3D points and their corresponding weights.

    Input:
    :param X: (numpy array of shape (N, 3)) A set of N 3D points (point coordinates).
    :param a: (numpy array of shape (N,)) A set of N weights corresponding to each point.
    :predefined param radius: *(number) Radius of the occlusion sphere

    Output:
    :return X_occluded: (numpy array) The points that remain after occlusion.
    :return a_occluded: (numpy array) The corresponding weights, renormalized to sum to 1.
    '''

    ## Center input points in [0,1^3]

    # Min-max normalization to [0, 1] along each column (x, y, z)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Avoid division by zero in case max == min
    X = (X - min_vals) / (max_vals - min_vals + 1e-8)

    # Define occlusion sphere
    sphere_center = np.array([0.5, 0.5, 0.5])  # Center of the occlusion

    # Compute distance from each point to the sphere center
    distances = np.linalg.norm(X - sphere_center, axis=1)

    # Mask points outside the sphere
    mask = distances > radius

    # Apply mask to X and a
    X_occluded = X[mask]  # Keep only points outside the occlusion sphere
    a_occluded = a[mask]  # Remove mass inside the sphere

    # Re-normalize `a` so it sums to 1
    a_occluded /= a_occluded.sum()

    return X_occluded, a_occluded



#number of sample points
num_points_to_sample = 500



## OCCLUSION FUNCTION IN ONE SAMPLE ###############################################################
print('First, let us visualize corruption of a 3D point cloud due to occlusion')

# Example category and sample file name
category = 'airplane'  # Replace with the desired category
sample_file = 'airplane_0236.off'  # Replace with the actual .off file

# Construct the full path to the .off file
sample_file_path = os.path.join(dataset_path, 'ModelNet40', category, 'train', sample_file)

# Load the mesh using trimesh
mesh = trimesh.load_mesh(sample_file_path)

# Sample points from the mesh surface (e.g., 1000 points)
sampled_points_example = mesh.sample(num_points_to_sample)

# Plot the sampled points in 3D using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_points_example[:, 0], sampled_points_example[:, 1], sampled_points_example[:, 2], s=1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Input point cloud")

ax.set_axis_off()
ax.grid(False)

plt.show()

# measure (uniform)
a = np.ones(num_points_to_sample) / num_points_to_sample
# Apply circular occlusion
X_occluded, a_occluded = occlusion_spherical(sampled_points_example, a)
a_occluded = a_occluded/a_occluded.sum()


# Plot the sampled points in 3D using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_occluded[:, 0], X_occluded[:, 1], X_occluded[:, 2], s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Corrupted input point cloud")

ax.set_axis_off()
ax.grid(False)

plt.show()

## Compute distance matrix of the occluded input example
input_matrix_occ = sp.spatial.distance.cdist(X_occluded, X_occluded)



print('In this experiment, we will reconstruct its perturbed version')


## GET TEMPLATES ##################################################################################
print('Get templates and their perturbations via occlusion')

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

# Store the sampled points for each airplane
sampled_airplanes = []
# list of dissimilarity matrices
matrix_temp_list = []
# list of measures
measure_temp_list = []


# list of occluded sample points
occluded_points = []
# list of template measures occluded
measure_occ_list = []
# list of template dissimilarity matrices occluded
matrix_occ_list = []

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

    # Apply occlusion to the spatial coordinates and probability measure
    P_occluded, p_occluded = occlusion_spherical(normalized_points, p_s)

    p_occluded = p_occluded/p_occluded.sum()
    measure_occ_list.append(p_occluded)

    occluded_points.append(P_occluded)

    # Compute the distance matrix for the occluded points
    dist_matrix_occ = sp.spatial.distance.cdist(P_occluded, P_occluded)
    matrix_occ_list.append(dist_matrix_occ)


# PLOT TEMPLATES
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


# PLOT OCCLUDED TEMPLATES
# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

# Set the main title for the entire figure
fig.suptitle("Corrupted Templates", fontsize=16)

# Plot each normalized airplane in a separate subplot
for i, (ax, o_p) in enumerate(zip(axes, occluded_points[:3])):  # Use first 3 airplanes
    ax.scatter(o_p[:, 0], o_p[:, 1], o_p[:, 2], s=1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Airplane {i + 1}")

    ax.set_axis_off()
    ax.grid(False)

plt.tight_layout()

plt.show()




## SOLVE ANALYSIS PROBLEM IN THE OCCLUSION WORLD ##################################################
print('Solving the GW-analysis problem')
_, lambdas = utils.get_lambdas(matrix_occ_list, measure_occ_list, input_matrix_occ, a_occluded)
print('GW-analysis problem, solved')


## Reconstruct
print('Now, reconstruct by solving a GW-synthesis problem via POT pre-defined functions')
M = 300
b = np.ones(M)/M
B_recon = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas)
B_recon = (B_recon + B_recon.T) / 2  # sym

## Compare Original target vs reconstruction
original_dist_mat = sp.spatial.distance.cdist(sampled_points_example, sampled_points_example)
N_original = len(sampled_points_example)
original_measure = np.ones(N_original)/N_original
gromov_distance = ot.gromov.gromov_wasserstein(original_dist_mat, B_recon, original_measure, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')


# Create an MDS instance
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)

## Fit and transform the distance matrix of B_recon
points_recon = mds.fit_transform(B_recon)

## Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_recon[:, 0], points_recon[:, 1], points_recon[:, 2], s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Reconstructed point cloud")

ax.set_axis_off()
ax.grid(False)

plt.show()




## Now with synthetic data
print('Repeating the experiment but for synthetic data')

## Get vector of weights
#lambdas_list = np.random.rand(n_temp)
#lambdas_list = lambdas_list/lambdas_list.sum()
lambdas_list =  np.random.dirichlet(np.ones(n_temp), size=1)[0]


#Synthesize a Barycenter using POT
M = 300 # Dimension of output barycentric matrix is MxM.

b = np.ones(M) / M   # Uniform target probability vector
B =  ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas_list)  # Synthesize barycenter matrix

# convert matrix to point cloud
points = mds.fit_transform(B_recon)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Synthetic Barycenter")

ax.set_axis_off()
ax.grid(False)

plt.show()


# occlude
# Apply occlusion to the spatial coordinates and probability measure
B_occluded, b_occluded = occlusion_spherical(points, b)
b_occluded = b_occluded/b_occluded.sum()

# Compute the distance matrix for the occluded points
input_matrix_occ_new = sp.spatial.distance.cdist(B_occluded, B_occluded)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(B_occluded[:, 0], B_occluded[:, 1], B_occluded[:, 2], s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Synthetic Barycenter Corrupted")

ax.set_axis_off()
ax.grid(False)

plt.show()


## SOLVE ANALYSIS PROBLEM IN THE OCCLUSION WORLD ##################################################
print('Solving the GW-analysis problem')
_, lambdas = utils.get_lambdas(matrix_occ_list, measure_occ_list, input_matrix_occ_new, b_occluded)
print('GW-analysis problem, solved')


## Reconstruct
print('Now, reconstruct by solving a GW-synthesis problem via POT pre-defined functions')
b = np.ones(M)/M
B_recon = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas)
B_recon = (B_recon + B_recon.T) / 2  # sym

## Compare Original target vs reconstruction
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')

## Fit and transform the distance matrix of B_recon
points_recon = mds.fit_transform(B_recon)

## Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_recon[:, 0], points_recon[:, 1], points_recon[:, 2], s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title(f"Synthetic Barycenter Recovered")

ax.set_axis_off()
ax.grid(False)

plt.show()

## Print lambda-vectors: original, after analysis, error
print('Original lambda-vector = ', lambdas_list)
print('Recovered lambda-vector = ', lambdas)
print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))