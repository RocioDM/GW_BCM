import numpy as np
import ot


def get_input(M):
    D = np.random.rand(M, M)
    D = D.T + D
    np.fill_diagonal(D, 0)
    return D

def get_template(s):
    N = 7
    C = np.random.rand(N,N)
    C = C.T + C
    np.fill_diagonal(C, 0)
    p = np.ones(C.shape[0]) / C.shape[0]
    return C, p

def main(i):
    np.random.seed(i)
    S = 5 # Number of templates or vertices
    M = 6 # Dimension of input and output matrices is  MxM.



    p_list = []
    C_list = []

    for s in range(S):
        C_s, p_s = get_template(s)
        p_list.append(p_s)
        C_list.append(C_s)

    #D = get_input(M)
    lambdas_list = np.random.rand(S)
    lambdas_list = lambdas_list/lambdas_list.sum()
    q = np.ones(M) / M
    D =  ot.gromov.gromov_barycenters(M, C_list, p_list,q, lambdas_list)



    pi_list = []
    F_list = []
    Q = (q.reshape(-1,1) @ q.reshape(1,-1))
    Q = 1. / Q

    for s in range(S):
        pi_s = ot.gromov.gromov_wasserstein(C_list[s],D,p_list[s],q)
        pi_list.append(pi_s)
        F_s = Q * (pi_s.T @ C_list[s] @ pi_s)
        F_list.append(F_s)

    print(f'F_list: {F_list[1]}')

    '''
    K = np.zeros((S,S))
    b = np.zeros(S)
    for i in range(S):
        b[i] = np.trace(D @ F_list[i])
        for j in range(S):
            K[i,j] = np.trace(F_list[i] @ F_list[j])

    lambdas = np.linalg.solve(K, b)
    print(lambdas.sum())
    '''


    K = np.zeros((S,S))
    b = np.zeros(S)
    for i in range(S):
        b[i] = np.trace(D @ F_list[i])
        for j in range(S):
            K[i,j] = np.trace(F_list[i] @ F_list[j])

    K_aug = np.hstack([K, -0.5 * np.ones(S).reshape(-1,1)])
    last_row = np.ones(K_aug.shape[1]).reshape(1,-1)
    last_row[0,-1] = 0
    K_aug = np.vstack([K_aug,last_row])

    b_aug = np.hstack([b, [1]])
    #print(b_aug)
    #print(K_aug)

    lambdas = np.linalg.solve(K_aug, b_aug)
    #print(f'Lambdas: {lambdas}')
    #print(lambdas[0:-1].sum())

    lambda_indices = [lambdas[0:-1] < 0]
    #print(f'lambdas list: {lambdas_list}')
    #print(f'lambdas computed: {lambdas}')
    print(np.linalg.norm(lambdas_list - lambdas[0:-1]))


    # Reconstruct D_lamba_list barra
    D_recon = np.zeros_like(D)
    for s in range(S):
        D_recon += lambdas[s] * F_list[s]

    # Compare D vs D_barra
    print(f'D max:{abs(D).max()}')
    print(f'D-D_recon: {abs((D-D_recon)).max()}')
    print(f'D: {D}')
    print(f'D_recon: {D_recon}')


    return lambdas[0:-1].min()







    print('ready')



if __name__ == '__main__':
    '''
        This code checks if $\overline{D_\lambda} = D_\lambda$. Which was proven in the overleaf. 
        The experiments coincide with the theory. 
    '''
    for i in range(0,1):
        if main(i)>=0:
            print(i)
    #Good seeds = [13,15,21,29]

