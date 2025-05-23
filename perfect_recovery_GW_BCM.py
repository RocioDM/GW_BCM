## 3D Data + blow-up

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
from sklearn.manifold import MDS
import random

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


## GET TEMPLATES
#number of templates
n_temp = 3 # For this experiment we use exactly 3 templates due to the visualization

# List of 5 different airplane sample files
airplane_files = [
    'airplane_0236.off',
    'airplane_0435.off',
    'airplane_0215.off',
]

## Bounds for sample points from the mesh surface
# l_bound = 200
# u_bound = 500

# List of desired number of samples for each airplane
sample_sizes = [200, 200, 200]


# Store the sampled points for each airplane
sampled_airplanes = []
# list of dissimilarity matrices
matrix_temp_list = []
# list of measures
measure_temp_list = []

## Loop through each airplane file and sample points
#for airplane_file in airplane_files:
for airplane_file, num_points_to_sample in zip(airplane_files, sample_sizes):
    # Construct the full path to the .off file
    sample_file_path = os.path.join(dataset_path, 'ModelNet40', 'airplane', 'train', airplane_file)

    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(sample_file_path)

    ##Random number of samples
    #num_points_to_sample = random.randint(l_bound, u_bound)

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



###################################################################################################

n_experiments = 10

T = np.array([[-1,0],[0,3**(1/2)],[1,0]])

color = [
    "#1f77b4",  # blue
    "#ff7f0e"  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf"   # cyan
]


lambdas_list = np.random.dirichlet([1,1,1],n_experiments)
recovered_lambdas_list_fp = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu = np.zeros_like(lambdas_list)


## Experiments with synthesized barycenters via POT
print('Starting experiments with synthesized barycenters via POT')

M = 200  # Dimension of output barycentric matrix is MxM.
b = np.ones(M) / M  # Uniform target probability vector

for i in range(n_experiments):
    print(f'Synthesizing barycenter for experiment {i+1}')
    B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas_list[i])  # Synthesize barycenter matrix

    print(f'Solving the GW-analysis problem from fixed point approach for experiment {i+1}')
    _, lambdas_fix_point = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
    recovered_lambdas_list_fp[i] = lambdas_fix_point

    print(f'Solving the GW-analysis problem from gradient approach for experiment {i+1}')
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    _, lambdas_blow_up = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)
    recovered_lambdas_list_bu[i] = lambdas_blow_up

## PLOT

# for the triangle
x = [-1, 0, 1, -1]
y = [0, np.sqrt(3), 0, 0]

# Create a figure with two subplots in one row
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for recovered_lambdas_list_fp
for i in range(n_experiments):
    original_lambda = lambdas_list[i].reshape(3, 1)
    recovered_lambda_fp = recovered_lambdas_list_fp[i].reshape(3, 1)

    original_point = (original_lambda * T).sum(axis=0)
    recovered_point_fp = (recovered_lambda_fp * T).sum(axis=0)

    axes[0].scatter(original_point[0], original_point[1], c='b')
    axes[0].scatter(recovered_point_fp[0], recovered_point_fp[1], c='r', marker='x')

axes[0].plot(x, y, 'b-', linewidth=1)
axes[0].fill(x[:-1], y[:-1], 'skyblue', alpha=0)
axes[0].set_xlim(-1.2, 1.2)
axes[0].set_ylim(-0.2, np.sqrt(3) + 0.2)
axes[0].axis('off')
axes[0].set_aspect('equal')
axes[0].set_title('Fixed Point Method')

# Plot for recovered_lambdas_list_bu
for i in range(n_experiments):
    original_lambda = lambdas_list[i].reshape(3, 1)
    recovered_lambda_bu = recovered_lambdas_list_bu[i].reshape(3, 1)

    original_point = (original_lambda * T).sum(axis=0)
    recovered_point_bu = (recovered_lambda_bu * T).sum(axis=0)

    axes[1].scatter(original_point[0], original_point[1], c='b')
    axes[1].scatter(recovered_point_bu[0], recovered_point_bu[1], c='r', marker='x')

# Plot the triangle for the second plot
axes[1].plot(x, y, 'b-', linewidth=1)
axes[1].fill(x[:-1], y[:-1], 'skyblue', alpha=0)
axes[1].set_xlim(-1.2, 1.2)
axes[1].set_ylim(-0.2, np.sqrt(3) + 0.2)
axes[1].axis('off')
axes[1].set_aspect('equal')
axes[1].set_title('Gradient Method via Blow-up')

plt.tight_layout()
plt.show()


## Experiments with synthesized barycenters via blow-up
print('Starting experiments with synthesized barycenters via blow-up')

recovered_lambdas_list_fp = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu = np.zeros_like(lambdas_list)

_, measure_a, temp_blow_up_a = utils.blow_up(matrix_temp_list, measure_temp_list, matrix_temp_list[1], measure_temp_list[1])

M = len(measure_a)  # Dimension of output barycentric matrix is MxM.
b = np.ones(M) / M  # Uniform target probability vector

for i in range(n_experiments):
    print(f'Synthesizing barycenter for experiment via blow-up{i+1}')
    l = lambdas_list[i]
    B = sum(l[j] * temp_blow_up_a[j] for j in range(n_temp))  # Synthesize barycenter matrix

    print(f'Solving the GW-analysis problem from fixed point approach for experiment {i+1}')
    _, lambdas_fix_point = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, measure_a)
    recovered_lambdas_list_fp[i] = lambdas_fix_point

    print(f'Solving the GW-analysis problem from gradient approach for experiment {i+1}')
    #B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    _, lambdas_blow_up = utils.get_lambdas_blowup(temp_blow_up_a, B, measure_a)
    recovered_lambdas_list_bu[i] = lambdas_blow_up

## PLOT

# for the triangle
x = [-1, 0, 1, -1]
y = [0, np.sqrt(3), 0, 0]

# Create a figure with two subplots in one row
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for recovered_lambdas_list_fp
for i in range(n_experiments):
    original_lambda = lambdas_list[i].reshape(3, 1)
    recovered_lambda_fp = recovered_lambdas_list_fp[i].reshape(3, 1)

    original_point = (original_lambda * T).sum(axis=0)
    recovered_point_fp = (recovered_lambda_fp * T).sum(axis=0)

    axes[0].scatter(original_point[0], original_point[1], c='b')
    axes[0].scatter(recovered_point_fp[0], recovered_point_fp[1], c='r', marker='x')

axes[0].plot(x, y, 'b-', linewidth=1)
axes[0].fill(x[:-1], y[:-1], 'skyblue', alpha=0)
axes[0].set_xlim(-1.2, 1.2)
axes[0].set_ylim(-0.2, np.sqrt(3) + 0.2)
axes[0].axis('off')
axes[0].set_aspect('equal')
axes[0].set_title('Fixed Point Method')

# Plot for recovered_lambdas_list_bu
for i in range(n_experiments):
    original_lambda = lambdas_list[i].reshape(3, 1)
    recovered_lambda_bu = recovered_lambdas_list_bu[i].reshape(3, 1)

    original_point = (original_lambda * T).sum(axis=0)
    recovered_point_bu = (recovered_lambda_bu * T).sum(axis=0)

    axes[1].scatter(original_point[0], original_point[1], c='b')
    axes[1].scatter(recovered_point_bu[0], recovered_point_bu[1], c='r', marker='x')

# Plot the triangle for the second plot
axes[1].plot(x, y, 'b-', linewidth=1)
axes[1].fill(x[:-1], y[:-1], 'skyblue', alpha=0)
axes[1].set_xlim(-1.2, 1.2)
axes[1].set_ylim(-0.2, np.sqrt(3) + 0.2)
axes[1].axis('off')
axes[1].set_aspect('equal')
axes[1].set_title('Gradient Method via Blow-up')

plt.tight_layout()
plt.show()