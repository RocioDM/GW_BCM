## The synthesis barycenter problem via different approaches
## This notebook visualize different interpolations between two MNIST point clouds
## Classical OT Interpolation - Euclidean interpolation - GW interpolation via barycenters using POT and blow-up

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

import ot  # POT: Python Optimal Transport Library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()

## Select two point clouds of digits (E.g., 0 and 1)
ind1=10
ind2=5

a = Data[ind1,:,2]
X = Data[ind1,a!=-1,:2]
X = X-X.mean(0)[np.newaxis,:]

b = Data[ind2,:,2]
Y = Data[ind2,b!=-1,:2]
Y = Y-Y.mean(0)[np.newaxis,:]

a = a[a!=-1]
b = b[b!=-1]

a = a/float(a.sum())
b = b/float(b.sum())



# Aesthetic parameters
fs = 18        # Font size for title
scale = 500    # Scaling factor for marker size
offset = 55    # Shift factor



# Create a figure with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(2, 1, figsize=(18, 6))

# interpolated points (1-t, t)
ts = np.linspace(0, 1, 7)




## OT Interpolation (first subplot)

# Compute cost matrix and optimal transport plan
C = ot.dist(X, Y)

Gamma = ot.emd(a, b, C)
GAMMA = Gamma.reshape(-1)

# Prepare data for interpolation
XX = np.stack([X]*Y.shape[0], axis=1).reshape(-1, 2)
DXX = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]).reshape(-1, 2)

for i, t in enumerate(ts):
    Xhat = XX - t * DXX
    shifted_x = Xhat[:, 0] + (1 - t) * (-offset) + t * offset
    axes[0].scatter(shifted_x, Xhat[:, 1], s=GAMMA * scale, c='C0', alpha=0.7)

    # Compute average x-position to place label
    avg_x = np.mean(shifted_x)
    axes[0].text(avg_x, -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

axes[0].set_ylim(-20, 15)
axes[0].set_title('OT-Barycenter Space via OT-Interpolation', fontsize=fs)
axes[0].set_xticks([])  # Remove x ticks
axes[0].set_yticks([])  # Remove y ticks
axes[0].set_aspect('equal', adjustable='box')




## Euclidean Interpolation (second subplot)

n= len(a)
m=len(b)

# Parameters
num_samples = 300

for i, t in enumerate(ts):
    if t == 0:
        samples = X
        sizes = a * scale
    elif t == 1:
        samples = Y
        sizes = b * scale
    else:
        samples = []
        sizes = []
        # Sample Bernoulli(t)
        for _ in range(num_samples):
            if np.random.rand() < t:
                idx = np.random.choice(m, p=b)
                samples.append(Y[idx])
                weight = (1 - t) * 0 + t * b[idx]
            else:
                idx = np.random.choice(n, p=a)
                samples.append(X[idx])
                weight = (1 - t) * a[idx] + t * 0
            sizes.append(weight * scale)
        samples = np.array(samples)
        sizes = np.array(sizes)

    shifted_x = samples[:, 0] + (1 - t) * (-offset) + t * offset
    axes[1].scatter(shifted_x, samples[:, 1], s=sizes, alpha=0.7, c='C0')

    avg_x = np.mean(shifted_x)
    axes[1].text(avg_x, -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

axes[1].set_ylim(-20, 15)
axes[1].set_xticks([])  # Remove x ticks
axes[1].set_yticks([])  # Remove y ticks
axes[1].set_title('Linear Barycenter Space via Convex Combinations', fontsize=fs)
axes[1].set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save the combined figure
plt.savefig("OT_and_Euclidean_interpolation.pdf", bbox_inches='tight')
plt.show()





## GW - Interpolation via Barycenters

## Dissimilarity matrices
C1 = ot.dist(X, X)
C2 = ot.dist(Y, Y)

# MDS embedding
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)


## GW - Interpolation via Barycenters using POT / Fix-point approach for computing GW-barycenters (separate plot)

## Parameters for the POT function ot.gromov.gromov_barycenters
M = max (len(a),len(b))
p = np.ones(M) / M

fig, ax = plt.subplots(figsize=(16, 5))

for i, t in enumerate(ts):
    if t == 0:
        Z = X
        Z = Z/10
        sizes = a * scale

    elif t == 1:
        Z = Y
        Z = Z/10
        sizes = b * scale

    else:
        # Compute GW barycenter distance matrix
        Cb = ot.gromov.gromov_barycenters(
            N=M, Cs=[C1, C2], ps=[a, b], p=p,
            lambdas=[1 - t, t], loss_fun='square_loss',
            symmetric=True, max_iter=1000, tol=1e-09
        )
        #sizes = ((1 - t) * np.interp(range(M), np.linspace(0, M, len(a)), a) + t * np.interp(range(M), np.linspace(0, M, len(b)), b)) * scale #Optional (aesthetic)
        sizes = 3*np.ones(M)
        # MDS embedding
        if t < 0.5:
            Z = mds.fit_transform(Cb,X)
        else:
            Z = mds.fit_transform(Cb,Y)

        #Center
        Z = Z - Z.mean(0)[np.newaxis, :]
        # Re-scale
        Z_min = Z.min(axis=0)
        Z_max = Z.max(axis=0)
        Z = Z / Z_max
        Z = Z

    # Shift points in x for visualization
    Z[:, 0] = Z[:, 0] + (1 - t) * (-7) + t * 7
    # Plot
    ax.scatter(Z[:, 0], Z[:, 1], s=sizes, alpha=0.7, c='C0')
    avg_x = np.mean(Z[:, 0])
    ax.text(avg_x, -1.5, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

# Styling
ax.set_ylim(-2, 1.5)
ax.set_xlim(-8.5, 8.5)
ax.set_title('GW Interpolation via Barycenters', fontsize=fs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("gw_interpolation.pdf", bbox_inches='tight')
plt.show()




## GW - Interpolation via Barycenters using the blow-up technique (separate plot)

# BLOW UP

# GW transport plan
pi = ot.gromov.gromov_wasserstein(C1, C2, a, b)

row_indices, col_indices = np.nonzero(pi)
non_zero_coords = np.array(list(zip(row_indices, col_indices)))
v_x = non_zero_coords[:, 0]
v_y = non_zero_coords[:, 1]
b_bu = pi[v_x, v_y]

V_1, V_2 = np.meshgrid(v_y, v_y)
YY = C2[V_1, V_2]

A_1, A_2 = np.meshgrid(v_x, v_x)
XX = C1[A_1, A_2]


# Interpolation and plot
fig, ax = plt.subplots(figsize=(16, 5))

for i, t in enumerate(ts):
    if t == 0:
        Z = X
        Z = Z/10
        sizes = a * scale

    elif t == 1:
        Z = Y
        Z = Z/10
        sizes = b * scale

    else:
        D_t = t * YY + (1 - t) * XX
        sizes = b_bu * scale
        # MDS embedding
        if t < 0.5:
            Z = mds.fit_transform(D_t,X)
        else:
            Z = mds.fit_transform(D_t,Y)

        #Center
        Z = Z - Z.mean(0)[np.newaxis, :]
        # Re-scale
        Z_min = Z.min(axis=0)
        Z_max = Z.max(axis=0)
        Z = Z / Z_max
        Z = Z

    # Shift points in x for visualization
    Z[:, 0] = Z[:, 0] + (1 - t) * (-7) + t * 7
    # Plot
    ax.scatter(Z[:, 0], Z[:, 1], s=sizes, alpha=0.7, c='C0')
    avg_x = np.mean(Z[:, 0])
    ax.text(avg_x, -1.5, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

# Styling
ax.set_ylim(-2, 1.5)
ax.set_xlim(-8.5, 8.5)
ax.set_title('GW Interpolation via Barycenters using blow up', fontsize=fs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("gw_interpolation_blow_up.pdf", bbox_inches='tight')
plt.show()


## Checking accuracy of the blow-up method (optional)
# gromov_distance = ot.gromov.gromov_wasserstein(XX, C1, b_bu, a, log=True)[1]
# gw_dist = gromov_distance['gw_dist']
# print(f'GW(X, X) : {gw_dist:.4f}')
#
# gromov_distance = ot.gromov.gromov_wasserstein(YY, C2, b_bu, b, log=True)[1]
# gw_dist = gromov_distance['gw_dist']
# print(f'GW(Y, Y) : {gw_dist:.4f}')
#
# gromov_distance = ot.gromov.gromov_wasserstein(XX, YY, b_bu, b_bu, log=True)[1]
# gw_dist = gromov_distance['gw_dist']
# print(f'GW(X, Y) : {gw_dist:.4f}')
#
# B = np.outer(b_bu, b_bu)
# # Compute the weighted Frobenius norm
# weighted_frob_norm = np.sum(B * (XX-YY)**2)
# print(f'Frob norm of difference X-Y after blow-up: {weighted_frob_norm}')
# print('Frob norm of X', np.linalg.norm(XX, 'fro'))
# print('Frob norm of Y', np.linalg.norm(YY, 'fro'))