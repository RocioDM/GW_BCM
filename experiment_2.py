import numpy as np
import ot
import matplotlib.pyplot as plt

def get_template(s):
    N = 7
    C = 100*np.random.rand(N,N)
    C = C.T + C
    np.fill_diagonal(C, 0)
    p = np.ones(C.shape[0]) / C.shape[0]
    return C, p

def main(i):
    np.random.seed(1)
    S = 5 # Number of templates or vertices
    M = 5 # Dimension of input and output matrices is  MxM.


    p_list = []
    C_list = []

    for s in range(S):
        C_s, p_s = get_template(s)
        p_list.append(p_s)
        C_list.append(C_s)

    q = np.ones(M) / M

    # Generate lambdas list
    lambdas_list = np.random.rand(S)
    lambdas_list = lambdas_list/lambdas_list.sum()

    # Compute barycenters for q, lambdas and p_s,C_s fixed using different seeds and POT
    np.random.seed(i)
    D_1 =  ot.gromov.gromov_barycenters(M, C_list, p_list,q, lambdas_list)

    np.random.seed(i+1)
    D_2 = ot.gromov.gromov_barycenters(M, C_list, p_list, q, lambdas_list)

    gromov_distance = ot.gromov_wasserstein(D_1,D_2,q,q,log=True)[1]
    #print(D_1)
    #print(D_2)

    return gromov_distance['gw_dist']




if __name__ == '__main__':
    '''
        Check that if we start with a fixed lambdas and C_s and compute the barycenters using Python OT with different
        seeds, then we get different barycenters.
        This is because the barycenter is not unique and the method used is iterative and depends on the initial guess.  
    '''
    results = []
    for i in range(0,100):
        results.append(main(i))
        #print(results[i])
    cut_numbers = [f"{num:.3e}" for num in results]
    abs_numbers = [f"{np.abs(float(num)):.3e}" for num in cut_numbers]
    diff_list = list(set(abs_numbers))
    print(diff_list)
    print(len(diff_list))

    #plt.plot(results,'.')
    #plt.show()
    #Good seeds = [13,15,21,29]

