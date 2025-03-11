#This notebook recovers the weights in the analysis problem of GW - barycenters and tests accuracy on pointcloud MNIST dataset.

import numpy as np  # linear algebra
import pandas as pd  # data processing
import scipy as sp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import time
import kagglehub
import ot

## USER DEFINED FUNCTIONS #########################################################################
def get_lambdas(matrix_temp_list, measure_temp_list, matrix_input, measure_input):
    '''
    Computes the barycentric weights (lambda_1,...,lambda_S), where S is the number of templates.
    Computes a candidate for a barycenter matrix associated with the barycenter weights using one iteration of the forward GW-barycenter problem.
      See for example remark 2.9 or equation 11.

    Input:
      matrix_temp_list: List of S template matrices (Ns x Ns) representing different dissimilarity matrices.
      measure_temp_list: List of S probability vectors (length Ns), representing probability measures of the S templates.
      matrix_input: (M x M) matrix representing the dissimilarity matrix to analyze.
      measure_input: Probability vector of length M.

    Output:
      lambdas: Vector of weights, one for each template (S elements). These are not necessarily non-negative.
      matrix_output: Synthesized dissimilarity matrix.
    '''
    S = len(matrix_temp_list)  # Number of template matrices
    pi_list = []  # List to store Gromov-Wasserstein transport plans
    F_list = []  # List to store transformed matrices
    # Compute Q matrix (inverse of the outer product of measure_input)
    Q = (measure_input.reshape(-1, 1) @ measure_input.reshape(1, -1))
    Q = 1. / Q  # Element-wise inverse

    # Compute Gromov-Wasserstein transport maps and one iteration of the forward GW-barycenter problem
    for s in range(S):
        # Compute optimal transport plan (pi_s) using Gromov-Wasserstein transport
        pi_s = ot.gromov.gromov_wasserstein(matrix_temp_list[s], matrix_input,
                                            measure_temp_list[s], measure_input)
        pi_list.append(pi_s)
        # Compute F_s transformation using Q and pi_s
        F_s = Q * (pi_s.T @ matrix_temp_list[s] @ pi_s)  # Element-wise multiplication
        F_list.append(F_s)

    # Set up and solve the linear problem (K@lambdas = b) for the vector of weights lambdas
    # Construct K matrix and b vector for least squares problem
    K = np.zeros((S, S))
    b = np.zeros(S)
    for i in range(S):
        b[i] = np.trace(matrix_input @ F_list[i])  # Compute b_i
        for j in range(S):
            K[i, j] = np.trace(F_list[i] @ F_list[j])  # Compute K_ij

    # Augment K with an additional row and column for sum-to-one constraint
    K_aug = np.hstack([K, -0.5 * np.ones(S).reshape(-1, 1)])
    last_row = np.ones(K_aug.shape[1]).reshape(1, -1)
    last_row[0, -1] = 0
    K_aug = np.vstack([K_aug, last_row])

    # Augment b with the constraint that lambdas sum to 1
    b_aug = np.hstack([b, [1]])
    # Solve for lambdas using the linear system K_aug * lambdas = b_aug
    lambdas = np.linalg.solve(K_aug, b_aug)
    # Extract lambda values (excluding the last auxiliary value corresponding to the lagrange multiplier of the sum-to-one constraint)
    lambdas = lambdas[:-1]
    # Compute the synthesized output matrix
    matrix_output = np.zeros_like(matrix_input)

    for s in range(S):
        matrix_output += lambdas[s] * F_list[s]  # Weighted sum of transformed matrices
        matrix_input = matrix_output  # Update matrix_input (although this might be redundant)

    return matrix_output, lambdas



## DATASET LOADING ################################################################################
# Download the dataset from Kaggle and save the path
path = kagglehub.dataset_download("cristiangarcia/pointcloudmnist2d")
# Load the test dataset from the downloaded files
df = pd.read_csv(path + "/test.csv")
# Extract numerical data (point cloud coordinates) from the dataset, excluding the first column (which contains labels)
Data = df[df.columns[1:]].to_numpy()
# Extract labels (digits) from the first column
label = df[df.columns[0]].to_numpy()
# Reshape data into an array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
Data = Data.reshape(Data.shape[0], -1, 3)
# Create a list of indices for each digit (0-9), grouping their occurrences in the dataset
digit_indices = [np.where(label == digit)[0].tolist() for digit in range(10)]



## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
digit = 3  # Pick a digit from 0 to 9
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
    C_s = C_s - C_s.mean(0)[np.newaxis, :]

    # Normalize coordinates to fit within the unit square [0,1]²
    C_s -= C_s.min(axis=0)  # Shift to start at 0
    C_s /= C_s.max(axis=0)  # Scale to fit within [0,1]

    # Compute the pairwise Euclidean distance matrix for C_s
    dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
    matrix_temp_list.append(dist_matrix_s)



## GENERATING RANDOM VECTOR OF WEIGHTS AND SYNTHESIZING A BARYCENTER USING POT ####################
# Random vector of weights
lambdas_list = np.random.rand(n_temp)
lambdas_list = lambdas_list / lambdas_list.sum()

# Synthesize a Barycenter using POT
M = 100  # Dimension of output barycentric matrix is MxM.

# b = np.ones(M) / M   # Uniform target probability vector
b = np.random.rand(M)
b = b / b.sum()  # Random target probability vector

B = ot.gromov.gromov_barycenters(M, matrix_temp_list, measure_temp_list, b,
                                 lambdas_list)  # Synthesize barycenter matrix
B = (B + B.T) / 2  # enforce symmetry of synthesized barycenter



## RECOVER VECTOR OF WEIGHTS AND RECONSTRUCTED BARYCENTER B_RECON FROM SYNTHESIZED BARYCENTER USING
# get_lambdas FUNCTION AND COMPUTING ERROR ########################################################
B_recon, lambdas = get_lambdas(matrix_temp_list, measure_temp_list, B, b)
print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))

## COMPARING POT-SYTHETIZED BARYCENTER B VS RECONSTRUCTED BARYCENTER B-RECON
# BY COMPUTING THE GW-DISTANCE BETWEEN THEM
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Target,Reconstructed Target): {gw_dist}')

# Enforce Symmetry of the Reconstructed Barycenter matrix
B_recon = (B_recon + B_recon.T) / 2



## PREPROCESS B AND B_RECON USING MDS TO VISUALIZE LATER ##########################################
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)  # Create an MDS instance

# Apply MDS embedding to Barycenter B and Reconstructed Barycenter B_recon,
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

# Center and fit points to be in the [0,1]x[0,1] square for visualization
points_B = points_B - points_B.mean(0)[np.newaxis, :]
points_B -= points_B.min(axis=0)
points_B /= points_B.max(axis=0)
points_B_recon = points_B_recon - points_B_recon.mean(0)[np.newaxis, :]
points_B_recon -= points_B_recon.min(axis=0)
points_B_recon /= points_B_recon.max(axis=0)



## PLOT TEMPLATES #################################################################################
fig, axes = plt.subplots(1, n_temp, figsize=(12, 6))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind, :, 2]
    X = Data[ind, a != -1, :2]
    X = X - X.mean(0)[np.newaxis, :]
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    a = a[a != -1]
    a = a / float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
    axes[i].set_title(f'Digit #{i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    # axes[i].axis('off')

plt.show()



## PLOT B AND B_RECON #############################################################################
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(points_B[:, 0], points_B[:, 1], s=b * 250)
axes[0].set_title('Input Barycenter')
axes[0].set_xticks([])  # Remove x-axis ticks
axes[0].set_yticks([])  # Remove y-axis ticks

axes[1].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 250)
axes[1].set_title('Reconstructed Barycenter')
axes[1].set_xticks([])  # Remove x-axis ticks
axes[1].set_yticks([])  # Remove y-axis ticks

plt.show()