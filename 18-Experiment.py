
## This note is meant to assess the "projection" aspect of our methods

## Three fixed input shapes are chosen (triangle, square, and cross) and then visualize the distance of the barycenters to the shape that should be projected, i.e. to plot the distance functional on the simplex and mark the result of your algorithms. Ideally, they should lie around the minimum. The gap between your analysis methods and the projection problem should be discussed more detailed, especially, if it cannot be solved in the present paper.


import os
from pathlib import Path

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn.manifold import MDS


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

## FIXED-POINT METHOD
#obj_recon, lambdas_obj = utils.get_lambdas(Cs_except_i, ps_except_i,Cs[i],ps[i])
obj_recon, lambdas_obj = utils.get_lambdas_constraints(Cs_except_i, ps_except_i,Cs[i],ps[i])


##BLOW-UP METHOD
# obj_recon, b, temp_blow_up = utils.blow_up(Cs_except_i, ps_except_i,Cs[i],ps[i])
# obj_recon, lambdas_obj = utils.get_lambdas_blowup(temp_blow_up, obj_recon,b)




# ## PLOT  ##########################################################################################
# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
#
# # 1) Reconstructed barycenter (from utils.get_lambdas)
# points_recon = mds.fit_transform(obj_recon)
#
# # 2) True star shape xs[3]
# points_obj = mds.fit_transform(Cs[i])
#
# # 3) Templates: MDS embedding for the 3 kept templates
# points_temp = [mds.fit_transform(Cs_except_i[s]) for s in range(S-1)]
#
# ## PLOT THE 3 TEMPLATES ##########################################################################
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
# for k in range(3):
#     axes[k].scatter(points_temp[k][:, 0], points_temp[k][:, 1], s=100)
#     axes[k].set_title(f'Template {k+1}')
#     axes[k].set_xticks([])
#     axes[k].set_yticks([])
#     axes[k].set_aspect('equal')
#     axes[k].set_frame_on(False)  # <- removes the box
#
# plt.tight_layout()
# plt.show()


# ## PLOT RECONSTRUCTED vs OBJECTIVE ############################################################
# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# # --- True Objective ---
# axes[0].scatter(points_obj[:, 0], points_obj[:, 1], s=100)
# axes[0].set_title('True Objective')
# axes[0].set_xticks([])
# axes[0].set_yticks([])
# axes[0].set_aspect('equal')
#
# # --- Reconstructed barycenter ---
# axes[1].scatter(points_recon[:, 0], points_recon[:, 1], s=100)
# axes[1].set_title('Projection onto Barycenter Space')
# axes[1].set_xticks([])
# axes[1].set_yticks([])
# axes[1].set_aspect('equal')
#
# plt.tight_layout()
# plt.show()


###################################################################################################
print('Comparing with real synthetic GW Barycenters')

## simplex grid --> triangle (3 templates)
N = 10
pts = []
for l in range(N + 1):
    for j in range(N + 1 - l):
        k = N - l - j
        pts.append((l / N, j / N, k / N))
pts=np.array(pts)

np.save("simplex_grid.npy", pts)


dist_list_target = []
dist_list_recon = []
dist_list_temp1 = []
dist_list_temp2 = []
dist_list_temp3 = []

h = 1
for point in pts:
    #if h not in [10]:
        B = ot.gromov.gromov_barycenters(M, Cs_except_i, ps_except_i, p,
                                         point, max_iter=5000, tol=1e-16)

        gw_dist_target = ot.gromov.gromov_wasserstein2(B, Cs[i], p, ps[i])
        gw_dist_recon = ot.gromov.gromov_wasserstein2(B, obj_recon, p, ps[i])
        gw_dist_temp1 = ot.gromov.gromov_wasserstein2(B, Cs[0], p, ps[0])
        gw_dist_temp2 = ot.gromov.gromov_wasserstein2(B, Cs[1], p, ps[1])
        gw_dist_temp3 = ot.gromov.gromov_wasserstein2(B, Cs[2], p, ps[2])

        dist_list_target.append(gw_dist_target)
        dist_list_recon.append(gw_dist_recon)
        dist_list_temp1.append(gw_dist_temp1)
        dist_list_temp2.append(gw_dist_temp2)
        dist_list_temp3.append(gw_dist_temp3)

        print(f'iteration {h} out of {len(pts)}')
        h = h+1

dist_target = np.array(dist_list_target)
dist_recon = np.array(dist_list_recon)
dist_temp1 = np.array(dist_list_temp1)
dist_temp2 = np.array(dist_list_temp2)
dist_temp3 = np.array(dist_list_temp3)

np.save("gw_distances_target.npy", dist_target)
np.save("gw_distances_recon.npy", dist_recon)
np.save("gw_distances_temp1.npy", dist_temp1)
np.save("gw_distances_temp2.npy", dist_temp2)
np.save("gw_distances_temp3.npy", dist_temp3)










