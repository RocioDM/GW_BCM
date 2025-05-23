import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


# # Path to the downloaded dataset
# dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub
#
#
# ## GET TEMPLATES
# #number of templates
# n_temp = 3
#
# # List of 5 different airplane sample files
# airplane_files = [
#     'airplane_0236.off',
#     'airplane_0435.off',
#     'airplane_0215.off',
#     #'airplane_0162.off',
#     #'airplane_0303.off',
# ]  # Replace with the actual files in your dataset
#
# # Bounds for sample points from the mesh surface (e.g., 1000 points)
# l_bound = 200
# u_bound = 500
#
# # Store the sampled points for each airplane
# sampled_airplanes = []
# # list of dissimilarity matrices
# matrix_temp_list = []
# # list of measures
# measure_temp_list = []
#
# # Loop through each airplane file and sample points
# for airplane_file in airplane_files:
#     # Construct the full path to the .off file
#     sample_file_path = os.path.join(dataset_path, 'ModelNet40', 'airplane', 'train', airplane_file)
#
#     # Load the mesh using trimesh
#     mesh = trimesh.load_mesh(sample_file_path)
#
#     #Random number of samples
#     num_points_to_sample = random.randint(l_bound, u_bound)
#
#     # Sample points from the mesh surface
#     sampled_points = mesh.sample(num_points_to_sample)
#
#     # Normalize the points to fit within [0, 1]^3
#     min_vals = sampled_points.min(axis=0)
#     max_vals = sampled_points.max(axis=0)
#     normalized_points = (sampled_points - min_vals) / (max_vals - min_vals)
#
#     # Append the normalized points to the list
#     sampled_airplanes.append(normalized_points)
#
#     # Dissimilarity matrices
#     dist_matrix = sp.spatial.distance.cdist(normalized_points, normalized_points)
#     matrix_temp_list.append(dist_matrix)
#
#     # Measure
#     p_s = np.ones(num_points_to_sample) / num_points_to_sample
#     measure_temp_list.append(p_s)
#
#
# B = matrix_temp_list[0]
# b = measure_temp_list[0]
# B, b, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
#
#
# # Create a figure with 3 subplots (1 row, 3 columns)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
#
# # Set the main title for the entire figure
# fig.suptitle("Templates", fontsize=16)
#
# # Plot each normalized airplane in a separate subplot
# for i, (ax, sampled_points) in enumerate(zip(axes, sampled_airplanes[:3])):  # Use first 3 airplanes
#     ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)
#
#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f"Airplane {i + 1}")
#
#     ax.set_axis_off()
#     ax.grid(False)
#
# plt.tight_layout()
#
# plt.show()
#
#
#
# # Create an MDS instance
# mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
#
# ## Fit and transform the distance matrix
# points_temp_bu = []
# for i in range(n_temp):
#     points = mds.fit_transform(temp_blow_up[i])
#     points_temp_bu.append(points)
#
#
# # Create a figure with 3 subplots (1 row, 3 columns)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
#
# for i, (ax, sampled_points) in enumerate(zip(axes, points_temp_bu)):  # Use first 3 airplanes
#     ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1)
#
#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f"Airplane Blow-up {i + 1}")
#
#     ax.set_axis_off()
#     ax.grid(False)
#
# plt.tight_layout()
#
# plt.show()


# Load dataset
Data, label, digit_indices = utils.load_pointcloudmnist2d()

# Select two point clouds of digits (e.g., 0 and 1)
ind1 = 10
ind2 = 5

a = Data[ind1, :, 2]
X = Data[ind1, a != -1, :2]
X_original = X - X.mean(0)[np.newaxis, :]

b = Data[ind2, :, 2]
Y = Data[ind2, b != -1, :2]
Y_original = Y - Y.mean(0)[np.newaxis, :]

a = a[a != -1]
b = b[b != -1]

a = a / float(a.sum())
b = b / float(b.sum())
c = b



# Compute dissimilarity matrices
X_dist = ot.dist(X_original, X_original)
Y_dist = ot.dist(Y_original, Y_original)




# BLOW UP

# GW transport plan
pi = ot.gromov.gromov_wasserstein(X_dist, Y_dist, a, b)

row_indices, col_indices = np.nonzero(pi)
non_zero_coords = np.array(list(zip(row_indices, col_indices)))
v_x = non_zero_coords[:, 0]
v_y = non_zero_coords[:, 1]
b = pi[v_x, v_y]

V_1, V_2 = np.meshgrid(v_y, v_y)
Y_tilde = Y_dist[V_1, V_2]
Y = Y_tilde.copy()

A_1, A_2 = np.meshgrid(v_x, v_x)
X_tilde = X_dist[A_1, A_2]
X = X_tilde.copy()


gromov_distance = ot.gromov.gromov_wasserstein(X, X_dist, b, a, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(X, X) : {gw_dist:.4f}')

gromov_distance = ot.gromov.gromov_wasserstein(X, Y, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(X, Y) : {gw_dist:.4f}')
B = np.outer(b, b)
# Compute the weighted Frobenius norm
weighted_frob_norm = np.sum(B * (X-Y)**2)
print(f'Frob norm of difference X-Y after blow-up: {weighted_frob_norm}')
print('Frob norm of X', np.linalg.norm(X, 'fro'))
print('Frob norm of Y', np.linalg.norm(Y, 'fro'))


# Interpolation and plotting
t_values = np.linspace(0, 1, 7)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

fig, axes = plt.subplots(2, len(t_values), figsize=(21, 7), constrained_layout=True)

label_y = -0.15 * max(np.ptp(X_original[:, 1]), np.ptp(Y_original[:, 1]))

for i, t in enumerate(t_values):
    if t == 0:
        embedding = X_original
        embedding_2 = X_original  # Optional: or skip
        size = a
        size_2 = a

    elif t == 1:
        embedding = Y_original
        embedding_2 = Y_original  # Optional: or skip
        size = c
        size_2 = c

    else:
        D_t = t * Y + (1 - t) * X
        N = 3*D_t.shape[0]
        B_t = ot.gromov.gromov_barycenters(N, [X_dist, Y_dist], [a, c], np.ones(N)/N, np.array([1 - t, t]), loss_fun='square_loss')
        gromov_distance = ot.gromov.gromov_wasserstein(D_t, B_t, b, np.ones(N)/N, log=True)[1]
        gw_dist = gromov_distance['gw_dist']
        print(f'GW(Blow_up_bary, POT_bary) at t={t:.2f}: {gw_dist:.4f}')
        print('Frob norm of Blow-up bary', np.linalg.norm(D_t, 'fro'))
        print('Frob norm of POT bary', np.linalg.norm(B_t, 'fro'))
        embedding = mds.fit_transform(D_t)
        embedding_2 = mds.fit_transform(B_t)
        size = b
        size_2 = np.ones(N)/N


    # Top row: Interpolated D_t
    ax = axes[0, i]
    ax.scatter(embedding[:, 0], embedding[:, 1], s=size * 300, c='C0', alpha=0.7)
    ax.set_aspect('equal')
    #ax.axis('off')
    ax.text(0.5, label_y, f'({1 - t:.2f}, {t:.2f})', transform=ax.transAxes,
            ha='center', va='top', fontsize=13)

    # Bottom row: POT barycenter B_t
    ax2 = axes[1, i]
    ax2.scatter(embedding_2[:, 0], embedding_2[:, 1], s= size_2*300, c='C1', alpha=0.7)
    ax2.set_aspect('equal')
    #ax2.axis('off')

# Titles
axes[0, 0].set_ylabel('Interpolation (Blow-up)', fontsize=14)
axes[1, 0].set_ylabel('POT Barycenter', fontsize=14)

fig.suptitle('Comparison: GW Interpolation vs POT Barycenters', fontsize=18)
plt.show()



