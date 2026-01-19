## This Notebook assess numerically sensitive of POT Library's function
## for generating GW barycenters

import numpy as np
import ot
from ot.gromov import gromov_barycenters, gromov_wasserstein2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

n_templates = 3
lam = np.ones(n_templates) / n_templates
n_experiments = 5

mean_distances = []
std_distances = []

Ns = [10, 20, 30, 40]
for N in Ns:
    # generate symmetric templates
    templates = []
    for _ in range(n_templates):
        ## Random symmetric matrix with zero-diag with entries between 0 and 1
        # A = 10*np.random.rand(N, N)
        # S = (A + A.T) / 2
        # np.fill_diagonal(S, 0)
        # S = S / np.linalg.norm(S, 'fro') #Normalize with Frob norm or np.max(S)

        ## Random distance
        A = np.random.rand(N,2)
        S = cdist(A,A)
        templates.append(S)

    templates_measures = [ot.unif(N) for _ in range(n_templates)]

    M = N
    q = ot.unif(M)

    distances = []

    rng = np.random.default_rng(42)

    for _ in range(n_experiments):
        state = rng.bit_generator.state
        np.random.seed(42)
        Y = gromov_barycenters(
            M,
            templates,
            templates_measures,
            q,
            lam,
            max_iter=1000,
            tol=1e-16
            )

        rng.bit_generator.state = state
        np.random.seed(42)
        Yp = gromov_barycenters(
            M,
            templates,
            templates_measures,
            q,
            lam,
            max_iter=1000,
            tol=1e-16
            )

        # GW distance between Y and Y'
        gw = gromov_wasserstein2(Y, Yp, ot.unif(M), ot.unif(M))
        distances.append(gw)

    mean_val = np.mean(distances)
    std_val = np.std(distances)

    mean_distances.append(mean_val)
    std_distances.append(std_val)

    print(f"{mean_val} ± {std_val}")

# Plot mean ± std as a function of N
plt.figure()
plt.errorbar(Ns, mean_distances, yerr=std_distances, marker='o', linestyle='-')
plt.xlabel("N (matrix size / number of points) of all templates and target")
plt.ylabel("GW distance (mean ± std)")
plt.title("GW distance between two identically generated barycenters vs N")
plt.grid(True)
plt.show()


