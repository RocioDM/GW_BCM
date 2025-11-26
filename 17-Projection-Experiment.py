## This notebook recovers the weights in the analysis problem of GW-barycenters
## and tests accuracy on PointCloud MNIST dataset (2D).
## In this notebook we use the function "get_lambdas" from "utils" (Fix-Point approach)

import numpy as np
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
digit = 5  # Pick a digit from 0 to 9
n_temp = 3  # Number of templates
ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices

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

    # Compute the pairwise Euclidean distance matrix for C_s
    dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
    matrix_temp_list.append(dist_matrix_s)


## GETTING A RANDOM POINT FROM DATASET ############################################################
print('Getting a Point from data set')
np.random.seed(42)
index = np.random.randint(len(digit_indices[digit]))
ind_obj = digit_indices[digit][index]
p_obj = Data[ind_obj, :, 2]
valid_indices = np.where(p_obj != -1)[0]
p_obj = p_obj[valid_indices]
p_obj = p_obj / float(p_obj.sum())
C_obj = Data[ind_obj, valid_indices, :2]
C_obj = utils.normalize_2Dpointcloud_coordinates(C_obj)
dist_matrix_obj = sp.spatial.distance.cdist(C_obj, C_obj)

## GW ANALYSIS: COORDINATES OF THE GIVEN POINT TO THE GIVEN TEMPLATES #############################
print('Estimating its GW-barycenter coordinates')
_, lambdas_obj = utils.get_lambdas(matrix_temp_list, measure_temp_list, dist_matrix_obj, p_obj)


###################################################################################################

print('Comparing with real synthetic GW Barycenters')

# Fix Size to Synthesize GW Barycenters using POT Library
M = 90  # Dimension of output barycentric matrix is MxM.
b = np.ones(M) / M   # Uniform probability vector


## simplex grid
N = 10
pts = []
for i in range(N + 1):
    for j in range(N + 1 - i):
        k = N - i - j
        pts.append((i / N, j / N, k / N))
pts=np.array(pts)


dist_list = []
h = 1
for point in pts:
    #if h not in [10]:
        B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                         point, max_iter=5000, tol=1e-16)

        gromov_distance = ot.gromov.gromov_wasserstein(B, dist_matrix_obj, b, p_obj, log=True)[1]
        gw_dist = gromov_distance['gw_dist']
        dist_list.append(gw_dist)
        print(f'iteration {h} out of {len(pts)}')
        h = h+1


# --- plot heatmap on simplex, with one black cross at special_pt (x,y,z) ---
def plot_simplex_heatmap(pts, vals, special_pt=None):
    # barycentric → 2D (equilateral triangle)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])

    xy = pts[:, 0, None] * A + pts[:, 1, None] * B + pts[:, 2, None] * C

    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c=vals, s=20)
    plt.colorbar()

    # draw simplex edges
    tri = np.array([A, B, C, A])
    plt.plot(tri[:, 0], tri[:, 1], 'k-')

    # optional black cross at special_pt in barycentric coords
    if special_pt is not None:
        special_pt = np.asarray(special_pt)
        sp_xy = special_pt[0] * A + special_pt[1] * B + special_pt[2] * C
        plt.scatter(sp_xy[0], sp_xy[1], c='k', marker='x', s=80)

    plt.axis('equal')
    plt.axis('off')
    plt.show()


special_pt = lambdas_obj #(1/3, 1/3, 1/3)

plot_simplex_heatmap(pts, dist_list, special_pt=special_pt)
