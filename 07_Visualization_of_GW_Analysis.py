## 3D Data
## This notebook recovers the weights in the analysis problem of GW-barycenters
## and visualizes them.
## GW-Barycenters are synthesized via POT and as convex combinations of the blow-up templates
## We use the functions "get_lambdas", "blow_up" and "get_lambdas_blowup" from "utils"
## We use 3 templates; we repeat the experiments "n_experiments" times;
## We visualize the true and estimated GW coordinates in the simplex,
## and analyze the recovery error.


import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
import random
import time
import math

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
l_bound = 30
u_bound = 30


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

n_experiments = 100



lambdas_list = np.random.dirichlet([1,1,1],n_experiments)


recovered_lambdas_list_fp = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu = np.zeros_like(lambdas_list)

# For POT-synthesized barycenters
times_fp_POT = np.zeros(n_experiments)
times_bu_POT = np.zeros(n_experiments)

## Experiments with synthesized barycenters via POT ###############################################
print('Starting experiments with synthesized barycenters via POT')

M = 30  # Dimension of output barycentric matrix is MxM.

## target probability vector:
b = np.ones(M) / M  # Uniform target probability vector

# b = np.random.rand(M)
# b = b / b.sum()  # Random target probability vector


compat_experiments_fp = 0 ## Compatible experiments

for i in range(n_experiments):
    print(f'Synthesizing barycenter for experiment {i+1}')

    B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b, lambdas_list[i], max_iter=5000, tol=1e-12)  # Synthesize barycenter matrix

    ###################################################################################################
    ## CHECK IF THIS SYNTHETIC BARYCENTER IS ACTUALLY A CRITICAL POINT FOR THE BLOW-UP APPROACH

    B_blowup, b_blowup, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)

    ## test if we have created a barycenter thru the blow-up method
    a = utils.get_lambdas_blowup_matrix(temp_blow_up, B_blowup, b_blowup)
    print(f'Barycenter test : {a}. If this value is zero, the element generated via POT Library is a critical point for the blow-up approach')  # if the value is zero, we have a barycenter

    if math.isclose(a, 0.0, rel_tol=1e-12, abs_tol=1e-15):
        compat_experiments_fp+=1
    ###################################################################################################

    print(f'Solving the GW-analysis problem from fixed point approach for experiment {i + 1}')
    t0 = time.perf_counter()
    _, lambdas_fix_point = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
    times_fp_POT[i] = time.perf_counter() - t0
    recovered_lambdas_list_fp[i] = lambdas_fix_point

    print(f'Solving the GW-analysis problem from gradient approach for experiment {i + 1}')
    t0 = time.perf_counter()
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    _, lambdas_blow_up = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)
    times_bu_POT[i] = time.perf_counter() - t0
    recovered_lambdas_list_bu[i] = lambdas_blow_up

    print(f'Blow-up size = {B_bu.shape}')



## Experiments with synthesized barycenters via blow-up ###########################################
print('Starting experiments with synthesized barycenters via blow-up')

recovered_lambdas_list_fp2 = np.zeros_like(lambdas_list)
recovered_lambdas_list_bu2 = np.zeros_like(lambdas_list)

# For blow-up-synthesized barycenters
times_fp_BU = np.zeros(n_experiments)
times_bu_BU = np.zeros(n_experiments)

_, measure_a, temp_blow_up_a = utils.blow_up(matrix_temp_list, measure_temp_list, matrix_temp_list[1], measure_temp_list[1])

M = len(measure_a)  # Dimension of output barycentric matrix is MxM.
b = np.ones(M) / M  # Uniform target probability vector

compat_experiments = 0

for i in range(n_experiments):
    print(f'Synthesizing barycenter for experiment via blow-up{i+1}')
    l = lambdas_list[i]
    B = sum(l[j] * temp_blow_up_a[j] for j in range(n_temp))  # Synthesize barycenter matrix

    ###################################################################################################
    ## CHECK IF THIS SYNTHETIC BARYCENTER IS ACTUALLY A CRITICAL POINT FOR THE BLOW-UP APPROACH

    B_blowup, b_blowup, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)

    ## test if we have created a barycenter thru the blow-up method
    a = utils.get_lambdas_blowup_matrix(temp_blow_up, B_blowup, b_blowup)
    print(
        f'Barycenter test : {a}. If this value is zero, the element generated via convex combinations of blow-up templates is a critical point for the blow-up approach')  # if the value is zero, we have a barycenter

    if math.isclose(a, 0.0, rel_tol=1e-12, abs_tol=1e-15):
     compat_experiments += 1

    ###################################################################################################

    print(f'Solving the GW-analysis problem from fixed point approach for experiment {i + 1}')
    t0 = time.perf_counter()
    _, lambdas_fix_point = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, measure_a)
    times_fp_BU[i] = time.perf_counter() - t0
    recovered_lambdas_list_fp2[i] = lambdas_fix_point

    print(f'Solving the GW-analysis problem from gradient approach for experiment {i + 1}')
    t0 = time.perf_counter()
    _, lambdas_blow_up = utils.get_lambdas_blowup(temp_blow_up_a, B, measure_a)
    times_bu_BU[i] = time.perf_counter() - t0
    recovered_lambdas_list_bu2[i] = lambdas_blow_up

    print(f'Blow-up size = {B_blowup.shape}')

print(f'Number of compatible experiments synthesize via POT Library: {compat_experiments_fp} out of {n_experiments}')

print(f'Number of compatible experiments synthesize as blow-up convex combinations of templates: {compat_experiments} out of {n_experiments}')


###################################################################################################
## PERFORMANCE METRICS: ERRORS AND VARIANCE #######################################################
###################################################################################################

# Errors for barycenters synthesized via POT
err_fp_POT = np.linalg.norm(recovered_lambdas_list_fp - lambdas_list, axis=1)
err_bu_POT = np.linalg.norm(recovered_lambdas_list_bu - lambdas_list, axis=1)

# rel_err_fp_POT = err_fp_POT / np.linalg.norm(lambdas_list, axis=1)

# Errors for barycenters synthesized via blow-up
err_fp_BU = np.linalg.norm(recovered_lambdas_list_fp2 - lambdas_list, axis=1)
err_bu_BU = np.linalg.norm(recovered_lambdas_list_bu2 - lambdas_list, axis=1)

print("===== POT-synthesized barycenters =====")
print(f"Fixed-point approach: mean error = {err_fp_POT.mean():.3e}, std = {err_fp_POT.std():.3e}")
print(f"Gradient via blow-up: mean error = {err_bu_POT.mean():.3e}, std = {err_bu_POT.std():.3e}")
print(f"Fixed-point time: mean = {times_fp_POT.mean():.3e} s, std = {times_fp_POT.std():.3e} s")
print(f"Blow-up time: mean = {times_bu_POT.mean():.3e} s, std = {times_bu_POT.std():.3e} s")

print("\n===== Blow-up-synthesized barycenters =====")
print(f"Fixed-point approach: mean error = {err_fp_BU.mean():.3e}, std = {err_fp_BU.std():.3e}")
print(f"Gradient via blow-up: mean error = {err_bu_BU.mean():.3e}, std = {err_bu_BU.std():.3e}")
print(f"Fixed-point time: mean = {times_fp_BU.mean():.3e} s, std = {times_fp_BU.std():.3e} s")
print(f"Blow-up time: mean = {times_bu_BU.mean():.3e} s, std = {times_bu_BU.std():.3e} s")


np.savez(
    "gw_3d_performance_results.npz",
    lambdas_true=lambdas_list,
    lambdas_fp_POT=recovered_lambdas_list_fp,
    lambdas_bu_POT=recovered_lambdas_list_bu,
    lambdas_fp_BU=recovered_lambdas_list_fp2,
    lambdas_bu_BU=recovered_lambdas_list_bu2,
    times_fp_POT=times_fp_POT,
    times_bu_POT=times_bu_POT,
    times_fp_BU=times_fp_BU,
    times_bu_BU=times_bu_BU,
    err_fp_POT=err_fp_POT,
    err_bu_POT=err_bu_POT,
    err_fp_BU=err_fp_BU,
    err_bu_BU=err_bu_BU,
)



## PLOT ###########################################################################################

# for the triangle
T = np.array([[-1,0],[0,3**(1/2)],[1,0]])
x = [-1, 0, 1, -1]
y = [0, np.sqrt(3), 0, 0]

fig, axes = plt.subplots(2, 2, figsize=(16, 16), gridspec_kw={'hspace': 0.35})

fig.suptitle("Synthesized GW Barycenters via Fixed-Point Iteration", fontsize=22, y=0.97)    ## i.e., via POT function
fig.text(0.5, 0.525, "Synthesized Convex Combinations of Blow-Up Templates",
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
axes[0, 1].set_title('Gradient Approach via Blow-Up', fontsize=20)
axes[1, 0].set_title('Fixed Point Approach', fontsize=20)
axes[1, 1].set_title('Gradient Approach via Blow-Up', fontsize=20)



# Save as PDF
plt.savefig("visualization_gw_barycenters_weights_analysis.pdf", format='pdf', bbox_inches='tight')

plt.show()