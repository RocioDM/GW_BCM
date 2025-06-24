## INTRODUCTION TO THE TUTORIAL:

## In this script we test the functions "get_lambdas" and "get_lambdas_constraints"
## from "utils"
## We generate random matrices that serve as templates;
## synthesize GW barycenters (for random coordinate vectors) using the POT Library;
## and we recover those coordinates.
## To analyze the performance of our methods, this process is repeated "n_experiments" times
## computing GW distances and differences between the true coordinate vectors and the estimated ones

import numpy as np
import ot  # POT: Python Optimal Transport library

from utils import get_lambdas, get_lambdas_constraints  # User-defined function for analysis step (fixed-point approach)


# Generate a random "template" distance matrix and a uniform probability distribution
def get_template(N=7):
    '''
    :param N: size of the template matrix (NxN) and probability vector (N)
    :return: symmetric matrix with zero diagonal (C) and probability vector (p)
    '''
    C = np.random.rand(N, N)
    C = C.T + C         #symmetric (optional)
    np.fill_diagonal(C, 0)      #zero-diagonal (optional)
    p = np.ones(C.shape[0]) / C.shape[0]  # Uniform distribution
    return C, p


def main(i, S=5, M=6):
    """
    Perform one trial of synthesis and analysis of a GW-barycenter.
    :param i: Random seed
    :param S: Number of templates (vertices)
    :param M: Dimension of input and output dissimilarity matrices (MxM)
    """
    np.random.seed(i)  # Reproducibility

    # Generate Templates
    p_list = []  # List of distributions for each template
    C_list = []  # List of distance matrices (templates)

    for s in range(S):
        C_s, p_s = get_template(N=np.random.randint(5, 10)) #templates with different sizes
        p_list.append(p_s)
        C_list.append(C_s)

    # Sample a random lambda (weights for the templates) and normalize to lie in simplex
    lambdas_list = np.random.dirichlet(np.ones(S), size=1)[0]  #random sample from the corresponding simplex

    # Define uniform distribution over target points
    q = np.ones(M) / M

    # Synthesize GW-barycenter based on random lambda
    D = ot.gromov.gromov_barycenters(M, C_list, p_list, q, lambdas_list)

    # Run analysis step: recover lambda weights from the synthesized barycenter (fixed-point approach)
    ## Testing the function get_lambdas
    D_recon, lambdas = get_lambdas(C_list, p_list, D, q)
    ## Uncomment the following line if you want to test the function get_lambdas_constraints
    #D_recon, lambdas = get_lambdas_constraints(C_list, p_list, D, q)

    ## Print lambda-vectors: original, recovered, and the L1 error between them
    print('Original lambda-vector = ', lambdas_list)
    print('Recovered lambda-vector (Fixed-Point approach) = ', lambdas)
    print('Lambdas Error = ', np.linalg.norm(lambdas_list - lambdas, 1))

    ## Compare original barycenter vs. reconstruction using GW distance
    gromov_distance = ot.gromov.gromov_wasserstein(D, D_recon, q, q, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Original, Reconstruction): {abs(gw_dist)}')

    return


# Run a series of experiments to test the analysis algorithm
if __name__ == '__main__':
    '''
    This script checks the performance of the analysis algorithm 
    using our function get_lambdas from the fix point approach.
    It check whether the recover lambdas coincide with the originals and 
    whether the recovered barycenter from analysis matches the one from synthesis
    '''
    n_experiments = 10
    for i in range(n_experiments):
        main(i)