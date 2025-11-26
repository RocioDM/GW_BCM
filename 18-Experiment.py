import os
from pathlib import Path

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn import manifold
from sklearn.decomposition import PCA

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils



## Data preparation
## The four distributions are constructed from 4 simple images (taken from POT tutorial)

def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


this_file = Path(__file__).resolve()
data_path = this_file.parent / "simple_shapes"

square = plt.imread(data_path / "square.png").astype(np.float64)[:, :, 2]
cross = plt.imread(data_path / "cross.png").astype(np.float64)[:, :, 2]
triangle = plt.imread(data_path / "triangle.png").astype(np.float64)[:, :, 2]
star = plt.imread(data_path / "star.png").astype(np.float64)[:, :, 2]

shapes = [square, cross, triangle, star]

S = 4
xs = [[] for i in range(S)]

for nb in range(4):
    for i in range(8):
        for j in range(8):
            if shapes[nb][i, j] < 0.95:
                xs[nb].append([j, 8 - i])

xs = [np.array(xs[s]) for s in range(S)]


ns = [len(xs[s]) for s in range(S)]
""""Compute all distances matrices for the four shapes"""
Cs = [sp.spatial.distance.cdist(xs[s], xs[s]) for s in range(S)]
Cs = [cs / cs.max() for cs in Cs]

ps = [ot.unif(ns[s]) for s in range(S)]

i = 3
Cs_except_i = [C for j, C in enumerate(Cs) if j != i]
ps_except_i = [p for j, p in enumerate(ps) if j != i]


M = 30
p = ot.unif(M)


###################################################################################################

print('Estimating its GW-barycenter coordinates')
obj_recon, lambdas_obj = utils.get_lambdas(Cs_except_i, ps_except_i,Cs[i],ps[i])




## PLOT RECONSTRUCTED BARYCENTER, AND OBJECTIVE USING MDS #######################
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

# 1) Reconstructed barycenter (from utils.get_lambdas)
points_recon = mds.fit_transform(obj_recon)

# 2) True star shape xs[3]
points_obj = mds.fit_transform(Cs[i])


## PLOT INPUT vs RECONSTRUCTED vs STAR ############################################################
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# --- True Objective ---
axes[0].scatter(points_obj[:, 0], points_obj[:, 1], s=100)
axes[0].set_title('True Objective')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_aspect('equal')

# --- Reconstructed barycenter ---
axes[1].scatter(points_recon[:, 0], points_recon[:, 1], s=100)
axes[1].set_title('Projection onto Barycenter Space')
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()




print('Comparing with real synthetic GW Barycenters')

## simplex grid --> triangle (3 templates)
N = 3
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
        B = ot.gromov.gromov_barycenters(M, Cs_except_i, ps_except_i, p,
                                         point, max_iter=5000, tol=1e-16)

        gromov_distance = ot.gromov.gromov_wasserstein(B, Cs[i], p, ps[i], log=True)[1]
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


special_pt = lambdas_obj

plot_simplex_heatmap(pts, dist_list, special_pt=special_pt)







