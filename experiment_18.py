## 3D Data

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils

# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


## GET TEMPLATES
#number of templates
n_temp = 3

# List of 5 different airplane sample files
airplane_files = [
    'airplane_0236.off',
    'airplane_0435.off',
    'airplane_0215.off'
    #'airplane_0162.off',
    #'airplane_0303.off',
]  # Replace with the actual files in your dataset

# Sample points from the mesh surface (e.g., 1000 points)
num_points_to_sample = 200

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
fig, axes = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': '3d'})

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


# Compute the Gromov-Wasserstein distances between different barycenters
# as M varies (i.e., support sizes 100, 200, ..., 600), to see how similar the barycenters are to each other.
# If the GW distances between barycenters with different M are small,
# it means your solution is stable across support sizes. Otherwise,
# it may indicate instability or sensitivity to the number of support points.

#Ms = list(range(100, 301, 100))  # [100, 200, ..., 600]
Ms = list(range(10, 500, 10))
B_list = []
b_list = []

# Step 1: Compute all barycenters for different support sizes
for M in Ms:
    print(f"Computing barycenter with M = {M}")
    b = np.ones(M) / M
    B = ot.gromov.gromov_barycenters(
        M,
        matrix_temp_list,
        measure_temp_list,
        b,
        lambdas_list,
        loss_fun='square_loss'
    )
    B_list.append(B)
    b_list.append(b)

# Step 2: Compute all pairwise GW distances between the barycenters
n = len(B_list)
gw_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i <= j:  # exploit symmetry
            _, log = ot.gromov.gromov_wasserstein(
                B_list[i], B_list[j],
                b_list[i], b_list[j],
                loss_fun='square_loss',
                log=True
            )
            gw_matrix[i, j] = gw_matrix[j, i] = log['gw_dist']
            if i != j:
                print(f'GW-distance {i,j} = {log['gw_dist']}')

# Step 3: Plot the pairwise GW distance matrix
plt.figure(figsize=(6, 5))
plt.imshow(gw_matrix, cmap='viridis')
plt.colorbar(label='GW Distance')
plt.xticks(ticks=range(n), labels=Ms)
plt.yticks(ticks=range(n), labels=Ms)
plt.title("Pairwise GW Distances Between Barycenters")
plt.xlabel("Barycenter M")
plt.ylabel("Barycenter M")
plt.show()

# Step 3: Set up the grid for plotting
ncols = 3  # Number of columns in the grid
nrows = (n * (n - 1)) // (2 * ncols) + 1  # Number of rows for the grid (upper triangular matrix)
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
axes = axes.flatten()  # Flatten axes to make indexing easier

# Step 4: Compute and plot GW couplings in grid format
plot_idx = 0  # To track subplot indexing
for i in range(n):
    for j in range(i, n):  # i ≤ j to avoid duplicates
        print(f"Computing T_gw between M={Ms[i]} and M={Ms[j]}")

        # Compute the GW coupling
        T_gw = ot.gromov.gromov_wasserstein(
            B_list[i], B_list[j],
            b_list[i], b_list[j],
            loss_fun='square_loss'
        )

        # Plot the GW coupling in the grid
        ax = axes[plot_idx]
        im = ax.imshow(T_gw, cmap='plasma', aspect='auto')
        ax.set_title(f'M={Ms[i]} → M={Ms[j]}')
        ax.set_xlabel(f'M={Ms[j]}')
        ax.set_ylabel(f'M={Ms[i]}')

        # Increment plot index for the next subplot
        plot_idx += 1

# Add colorbar to the last plot
fig.colorbar(im, ax=axes[-1], orientation='vertical', label='Coupling Value')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()