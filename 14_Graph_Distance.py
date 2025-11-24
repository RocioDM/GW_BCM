"""
This script computes approximate geodesic distances on a manifold represented by
a point cloud. The method first builds a k-nearest-neighbors (kNN) graph from the
input points, using Euclidean distances as edge weights. The geodesic distance
between two points is then approximated by the shortest-path distance along this
graph, computed via Dijkstra's algorithm.

New Functions:
----------
build_knn_graph(points, k):
    Constructs a weighted kNN graph where each point is connected to its k
    nearest neighbors.

pairwise_geodesic_distances(points, k):
    Computes the full pairwise geodesic distance matrix by running Dijkstra’s
    shortest-path algorithm from every point to all others.

This provides an approximation of manifold geodesic distances commonly used in
nonlinear dimensionality reduction techniques such as Isomap.
"""




import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import ot   # POT: Python Optimal Transport library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()

## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
print('Getting templates')
# Templates are of the form (matrix, measure)
digit = 2  # Pick a digit from 0 to 9
n_temp = 3  # Number of templates
ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices

###################################################################################################
###################################################################################################
def build_knn_graph(points, k=10):
    """
    Build a k-nearest neighbors graph from a point cloud.
    """
    N = points.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, indices = nbrs.kneighbors(points)

    G = nx.Graph()
    for i in range(N):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
            G.add_edge(i, j_idx, weight=dist)

    return G


def pairwise_geodesic_distances(points, k=10):
    """
    Compute the full pairwise geodesic distance matrix for a point cloud.

    Parameters
    ----------
    points : ndarray (N, d)
        Point cloud.
    k : int
        Number of neighbors for the kNN graph.

    Returns
    -------
    D : ndarray (N, N)
        Pairwise geodesic distance matrix.
    """
    G = build_knn_graph(points, k=k)
    N = points.shape[0]

    # Initialize matrix
    D = np.zeros((N, N))

    # Compute all-pairs shortest paths using Dijkstra
    for i in range(N):
        dist_dict = nx.single_source_dijkstra_path_length(G, i)
        for j, d_ij in dist_dict.items():
            D[i, j] = d_ij

    return D
###################################################################################################
###################################################################################################

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
    # and Normalize coordinates to fit within the unit square [0,1]x[0,1]
    C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

    # Compute the pairwise Graph distance matrix for C_s
    dist_matrix_s = pairwise_geodesic_distances(C_s)
    matrix_temp_list.append(dist_matrix_s)


## GENERATING RANDOM VECTOR OF WEIGHTS AND SYNTHESIZING A BARYCENTER USING POT ####################
print('Generating a GW-barycenter with a random lambda vector (synthesis through POT)')
# Random vector of weights
lambdas_list = np.random.dirichlet(np.ones(n_temp), size=1)[0]

# Synthesize a Barycenter using POT
M = 100  # Dimension of output barycentric matrix is MxM.

# b = np.ones(M) / M   # Uniform target probability vector
b = np.random.rand(M)
b = b / b.sum()  # Random target probability vector

B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 lambdas_list, max_iter=5000, tol=1e-16)  # Synthesize barycenter matrix
#B = (B + B.T) / 2  # Enforce symmetry of synthesized barycenter (optional)
#np.fill_diagonal(B, 0)      #zero-diagonal (optional)


## RECOVER VECTOR OF WEIGHTS AND RECONSTRUCTED BARYCENTER B_RECON FROM SYNTHESIZED BARYCENTER USING
# utils.get_lambdas FUNCTION AND COMPUTING ERROR ##################################################
print('Estimation of the lambda vector with our method, and reconstruction')
B_recon, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)
B_recon = (B_recon + B_recon.T) /2 # Enforce symmetry of the reconstructed barycenter matrix (optional)
np.fill_diagonal(B_recon, 0)      #zero-diagonal (optional)

print('Computing errors')

print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))

## COMPARING POT-SYNTHESIZED BARYCENTER B VS RECONSTRUCTED BARYCENTER B-RECON BY COMPUTING THE
# GW-DISTANCE BETWEEN THEM ########################################################################
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')



## PREPROCESS B AND B_RECON USING MDS TO VISUALIZE LATER ##########################################
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)  # Create an MDS instance

# Apply MDS embedding to Barycenter B and Reconstructed Barycenter B_recon,
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

# Center and fit points to be in the [0,1]x[0,1] square for visualization
points_B = utils.normalize_2Dpointcloud_coordinates(points_B)
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
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
    axes[i].set_title(f'Template {i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    # axes[i].axis('off')

plt.show()





## PLOT B AND B_RECON #############################################################################
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
axes[0].set_title('Input Barycenter')
axes[0].set_xticks([])  # Remove x-axis ticks
axes[0].set_yticks([])  # Remove y-axis ticks

axes[1].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
axes[1].set_title('Reconstructed Barycenter')
axes[1].set_xticks([])  # Remove x-axis ticks
axes[1].set_yticks([])  # Remove y-axis ticks


plt.show()