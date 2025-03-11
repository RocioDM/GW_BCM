import numpy as np  # linear algebra
import pandas as pd  # data processing
import scipy as sp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import time
import kagglehub
import ot

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