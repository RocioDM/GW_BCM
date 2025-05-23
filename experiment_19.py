##GW interpolation via barycenters

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


import os
import itertools

import ot  # Optimal transport library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils



## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()




## Select two indices for different point clouds
indices = [10, 5, 35, 30, 4, 52, 11, 17, 61, 9]


# Create output folder
output_folder = "figs_gw_interp"
os.makedirs(output_folder, exist_ok=True)

# Visualization parameters
fs = 18        # Font size for title
scale = 500    # Scaling factor for marker size
offset = 55    # Shift factor
ts = np.linspace(0, 1, 7)  # Interpolation points
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

# Loop over all combinations of positions in the list
for i0, i1 in itertools.combinations(range(len(indices)), 2):
    ind0 = indices[i0]
    ind1 = indices[i1]

    # Prepare first distribution
    a = Data[ind0, :, 2]
    X = Data[ind0, a != -1, :2]
    X = X - X.mean(0)[np.newaxis, :]
    a = a[a != -1]
    a = a / float(a.sum())

    # Prepare second distribution
    b = Data[ind1, :, 2]
    Y = Data[ind1, b != -1, :2]
    Y = Y - Y.mean(0)[np.newaxis, :]
    b = b[b != -1]
    b = b / float(b.sum())

    # Distance matrices and setup (same as before)...
    C1 = ot.dist(X, X)
    C2 = ot.dist(Y, Y)
    M = max(len(a), len(b))
    p = np.ones(M) / M

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, t in enumerate(ts):
        if t == 0:
            Z = X / 10
            sizes = a * scale
        elif t == 1:
            Z = Y / 10
            sizes = b * scale
        else:
            Cb = ot.gromov.gromov_barycenters(
                N=M, Cs=[C1, C2], ps=[a, b], p=p,
                lambdas=[1 - t, t], loss_fun='square_loss',
                symmetric=True, max_iter=1000, tol=1e-9
            )
            if t < 0.5:
                Z = mds.fit_transform(Cb, X)
            else:
                Z = mds.fit_transform(Cb, Y)
            Z = Z - Z.mean(0)[np.newaxis, :]
            Z = Z / Z.max(axis=0)
            sizes = 3 * np.ones(M)

        Z[:, 0] += (1 - t) * (-7) + t * 7
        ax.scatter(Z[:, 0], Z[:, 1], s=sizes, alpha=0.7, c='C0')
        ax.text(np.mean(Z[:, 0]), -1.5, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

    # Updated title and file name to reflect index positions
    ax.set_ylim(-2, 1.5)
    ax.set_xlim(-8.5, 8.5)
    ax.set_title(f'GW Interpolation: [{i0}] → [{i1}]', fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    filename = f"gw_interp_{i0}_{i1}.pdf"
    plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    plt.close(fig)
