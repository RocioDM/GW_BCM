## 3D Data
## This notebook recovers the weights in the analysis problem of GW-barycenters
## and visualizes them.
## GW-Barycenters are synthesized via POT and as convex combinations of the blow-up templates
## We use the functions "get_lambdas", "blow_up" and "get_lambdas_blowup" from "utils"
## We use 3 templates; we repeat the experiments "n_experiments" times;
## and visualize the true and estimated GW coordinates in the simplex.


import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
import random

import ot   # POT: Python Optimal Transport library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


## GET TEMPLATES

# List of 3 different airplane sample files
airplane_files = [
    'airplane_0236.off',
    'airplane_0435.off',
    'airplane_0215.off',
]
#number of templates
n_temp = len(airplane_files)


# Bounds for sample points from the mesh surface
l_bound = 200
u_bound = 400


# Store the sampled points for each airplane
sampled_data = []
# list of dissimilarity matrices
matrix_temp_list = []
# list of measures
measure_temp_list = []

## Loop through each airplane file and sample points
for airplane_file in airplane_files:
    # Construct the full path to the .off file
    sample_file_path = os.path.join(dataset_path, 'ModelNet40', 'airplane', 'train', airplane_file)

    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(sample_file_path)

    #Random number of samples
    num_points_to_sample = random.randint(l_bound, u_bound)
    print('template size = ', num_points_to_sample)
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



###################################################################################################

n_experiments = 10


lambdas_list = np.random.dirichlet([1,1,1],n_experiments)

recovered_lambdas_list_fp = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu = np.zeros_like(lambdas_list)


## Experiments with synthesized barycenters via POT ###############################################
print('Starting experiments with synthesized barycenters via POT')

M = 300  # Dimension of output barycentric matrix is MxM.
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




## Experiments with synthesized barycenters via blow-up ###########################################
print('Starting experiments with synthesized barycenters via blow-up')

recovered_lambdas_list_fp2 = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu2 = np.zeros_like(lambdas_list)

_, measure_a, temp_blow_up_a = utils.blow_up(matrix_temp_list, measure_temp_list, matrix_temp_list[1], measure_temp_list[1])

M = len(measure_a)  # Dimension of output barycentric matrix is MxM.
b = np.ones(M) / M  # Uniform target probability vector

for i in range(n_experiments):
    print(f'Synthesizing barycenter for experiment via blow-up{i+1}')
    l = lambdas_list[i]
    B = sum(l[j] * temp_blow_up_a[j] for j in range(n_temp))  # Synthesize barycenter matrix

    print(f'Solving the GW-analysis problem from fixed point approach for experiment {i+1}')
    _, lambdas_fix_point = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, measure_a)
    recovered_lambdas_list_fp2[i] = lambdas_fix_point

    print(f'Solving the GW-analysis problem from gradient approach for experiment {i+1}')
    #B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    _, lambdas_blow_up = utils.get_lambdas_blowup(temp_blow_up_a, B, measure_a)
    recovered_lambdas_list_bu2[i] = lambdas_blow_up




## PLOT ###########################################################################################

# for the triangle
T = np.array([[-1,0],[0,3**(1/2)],[1,0]])
x = [-1, 0, 1, -1]
y = [0, np.sqrt(3), 0, 0]

fig, axes = plt.subplots(2, 2, figsize=(16, 16), gridspec_kw={'hspace': 0.35})

fig.suptitle("Synthesized GW-Barycenters via Fixed Point Iteration", fontsize=22, y=0.97)    ## i.e., via POT function
fig.text(0.5, 0.525, "Synthesized GW-Barycenters via Combinations of Blow-up Templates",
         ha='center', va='top', fontsize=22)

for i in range(n_experiments):
    original_lambda = lambdas_list[i].reshape(3, 1)
    original_point = (original_lambda * T).sum(axis=0)

    recovered_lambda_fp = recovered_lambdas_list_fp[i].reshape(3, 1)
    recovered_point_fp = (recovered_lambda_fp * T).sum(axis=0)
    axes[0, 0].scatter(original_point[0], original_point[1], c='b')
    axes[0, 0].scatter(recovered_point_fp[0], recovered_point_fp[1], c='r', marker='x')

    recovered_lambda_bu = recovered_lambdas_list_bu[i].reshape(3, 1)
    recovered_point_bu = (recovered_lambda_bu * T).sum(axis=0)
    axes[0, 1].scatter(original_point[0], original_point[1], c='b')
    axes[0, 1].scatter(recovered_point_bu[0], recovered_point_bu[1], c='r', marker='x')

    recovered_lambda_fp2 = recovered_lambdas_list_fp2[i].reshape(3, 1)
    recovered_point_fp2 = (recovered_lambda_fp2 * T).sum(axis=0)
    axes[1, 0].scatter(original_point[0], original_point[1], c='b')
    axes[1, 0].scatter(recovered_point_fp2[0], recovered_point_fp2[1], c='r', marker='x')

    recovered_lambda_bu2 = recovered_lambdas_list_bu2[i].reshape(3, 1)
    recovered_point_bu2 = (recovered_lambda_bu2 * T).sum(axis=0)
    axes[1, 1].scatter(original_point[0], original_point[1], c='b')
    axes[1, 1].scatter(recovered_point_bu2[0], recovered_point_bu2[1], c='r', marker='x')

# Formatting plots
for ax_row in axes:
    for ax in ax_row:
        ax.plot(x, y, 'b-', linewidth=1)
        ax.fill(x[:-1], y[:-1], 'skyblue', alpha=0)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, np.sqrt(3) + 0.2)
        ax.axis('off')
        ax.set_aspect('equal')

axes[0, 0].set_title('Fixed Point Approach', fontsize=20)
axes[0, 1].set_title('Gradient Approach via Blow-up', fontsize=20)
axes[1, 0].set_title('Fixed Point Approach', fontsize=20)
axes[1, 1].set_title('Gradient Approach via Blow-up', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.91])

# Save as PDF
plt.savefig("visualization_gw_barycenters_weights_analysis.pdf", format='pdf', bbox_inches='tight')

plt.show()