## Classical OT Interpolation - GW interpolation - Euclidean interpolation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import scipy as sp
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS
from sklearn.manifold import smacof

import ot  # Optimal transport library

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


# Setup plot
fig, axes = plt.subplots(3, 1, figsize=(18, 6))

# interpolated points (1-t, t)
ts = np.linspace(0, 1, 7)

##########################
# 1. OT Interpolation
##########################
C = ot.dist(X, Y)
Gamma = ot.emd(a, b, C)
GAMMA = Gamma.reshape(-1)

XX = np.stack([X] * Y.shape[0], axis=1).reshape(-1, 2)
DXX = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]).reshape(-1, 2)

for i, t in enumerate(ts):
    Xhat = XX - t * DXX
    shifted_x = Xhat[:, 0] + (1 - t) * (-offset) + t * offset
    axes[0].scatter(shifted_x, Xhat[:, 1], s=GAMMA * scale, c='C0', alpha=0.7)

    avg_x = np.mean(shifted_x)
    axes[0].text(avg_x, -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

axes[0].set_ylim(-20, 15)
axes[0].set_title('OT-Barycenter Space via OT-Interpolation', fontsize=fs)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_aspect('equal', adjustable='box')

##########################
# 2. Euclidean Interpolation
##########################
n = len(a)
m = len(b)
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
        for _ in range(num_samples):
            if np.random.rand() < t:
                idx = np.random.choice(m, p=b)
                samples.append(Y[idx])
                weight = t * b[idx]
            else:
                idx = np.random.choice(n, p=a)
                samples.append(X[idx])
                weight = (1 - t) * a[idx]
            sizes.append(weight * scale)
        samples = np.array(samples)
        sizes = np.array(sizes)

    shifted_x = samples[:, 0] + (1 - t) * (-offset) + t * offset
    axes[1].scatter(shifted_x, samples[:, 1], s=sizes, alpha=0.7, c='C0')

    avg_x = np.mean(shifted_x)
    axes[1].text(avg_x, -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

axes[1].set_ylim(-20, 15)
axes[1].set_title('Linear Barycenter Space via Convex Combinations', fontsize=fs)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_aspect('equal', adjustable='box')

##########################
# 3. GW Interpolation via Barycenters
##########################
C1 = ot.dist(X, X)
C2 = ot.dist(Y, Y)
M = max(len(a), len(b))
p_uniform = np.ones(M) / M
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)

ax = fig.add_subplot(111, projection='3d')

for i, t in enumerate(ts):
    if t == 0:
        Z = X / 10
        sizes = a * scale
    elif t == 1:
        Z = Y / 10
        sizes = b * scale
    else:
        Cb = ot.gromov.gromov_barycenters(
            N=M, Cs=[C1, C2], ps=[a, b], p=p_uniform,
            lambdas=[1 - t, t], loss_fun='square_loss',
            symmetric=True, max_iter=1000, tol=1e-9
        )
        sizes = 3 * np.ones(M)
        Z_ref = X if t < 0.5 else Y
        Z = mds.fit_transform(Cb, Z_ref)
        Z = Z - Z.mean(0)[np.newaxis, :]
        Z = Z / Z.max(axis=0)

    # Horizontal offset
    Z[:, 0] += (1 - t) * (-7) + t * 7
    # Interpolation axis
    z = np.full_like(Z[:, 0], t)

    # Scatter plot in 3D
    ax.scatter(Z[:, 0], Z[:, 1], z, s=sizes, alpha=0.7, c='C0')

    # Label each interpolation step
    avg_x = np.mean(Z[:, 0])
    ax.text(avg_x, -1.6, 1.05, f'({1 - t:.2f}, {t:.2f})', fontsize=10, ha='center')

# Aesthetics
ax.set_title('GW-Barycenter Space (3D View)', fontsize=fs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Interpolation t')
ax.set_xlim(-8.5, 8.5)
ax.set_ylim(-2, 1.5)
ax.view_init(elev=30, azim=130)

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.savefig("combined_interpolations.pdf", bbox_inches='tight')
plt.show()











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
plt.savefig("combined_interpolation_space.pdf", bbox_inches='tight')
plt.show()




## GW - Interpolation via Barycenters
C1 = ot.dist(X, X)
C2 = ot.dist(Y, Y)


## Parameters for the POT function ot.gromov.gromov_barycenters
M = max (len(a),len(b))
p = np.ones(M) / M

# MDS embedding
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)


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
plt.savefig("gw_interpolation_better.pdf", bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4), facecolor='white')
scale = 200
for i, t in enumerate(ts):
    ax = axes[i]  # <-- each image gets its own subplot

    if t == 0:
        Z = X / 10
        sizes = a * scale
    elif t == 1:
        Z = Y / 10
        sizes = b * scale * 400
    else:
        # GW barycenter
        Cb = ot.gromov.gromov_barycenters(
            N=M, Cs=[C1, C2], ps=[a, b], p=p,
            lambdas=[1 - t, t], loss_fun='square_loss',
            symmetric=True, max_iter=1000, tol=1e-09
        )
        #sizes = ((1 - t) * np.interp(range(M), np.linspace(0, M, len(a)), a) + t * np.interp(range(M), np.linspace(0, M, len(b)), b)) * scale
        sizes = np.ones(M)
        if t < 0.5:
            Z = mds.fit_transform(Cb, X)
        else:
            Z = mds.fit_transform(Cb, Y)

        Z = Z - Z.mean(0)[np.newaxis, :]
        Z = Z / Z.max(0)

    # Shift is no longer needed since we have subplots
    # KDE
    kde = gaussian_kde(Z.T, weights=sizes / sizes.sum(), bw_method=0.12)
    xgrid = np.linspace(-2, 2, 300)
    ygrid = np.linspace(-2, 2, 300)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
    Z_kde = kde(positions).reshape(Xgrid.shape)
    Z_kde = Z_kde / Z_kde.max()
    Z_kde = Z_kde**0.5  # contrast boost

    #cmap = 'PuBu'
    cmap = 'gray_r'
    #cmap = 'bone'
    #cmap = 'Blues'
    ax.imshow(Z_kde, origin='lower', extent=[-1, 1, -1, 1],
              cmap=cmap, vmin=0, vmax=0.6)

    ax.text(avg_x-7, -1, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)
    #ax.set_title(f'({1 - t:.2f}, {t:.2f})', fontsize=12)
    ax.axis('off')

fig.suptitle('GW Barycenter', fontsize=18, y=0)
plt.tight_layout()
plt.savefig("gw_interpolation_kde.pdf", bbox_inches='tight', facecolor='white')
plt.show()

















# Compute GW optimal transport plan
gw_loss, log = ot.gromov.gromov_wasserstein2(C1, C2, a, b, 'square_loss', log=True)
T = log['T']

Gamma = T
GAMMA = Gamma.reshape(-1)

fig = plt.figure(figsize=(15, 4))

# Plot interpolated points and annotate with (1-t, t)
ts = np.linspace(0, 1, 7)
for i, t in enumerate(ts):
    Xhat = XX - t * DXX
    shifted_x = Xhat[:, 0] + (1 - t) * (-55) + t * 55
    plt.scatter(shifted_x, Xhat[:, 1], s=GAMMA * scale, c='C0', alpha=0.7)

    # Compute average x-position to place label
    avg_x = np.mean(shifted_x)
    plt.text(avg_x, -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)

# Aesthetics
plt.ylim(-20, 15)
plt.title('OT Barycenter Space', fontsize=fs)
plt.xticks([])  # Remove x ticks
plt.yticks([])  # Remove y ticks
plt.gca().set_aspect('equal', adjustable='box')

plt.show()  # Optional: still display




# # Parameters
# ts = np.linspace(0, 1, 7)
# fs = 18
# scale = 500
# offset = 70
# M = min(len(a), len(b))
# p = np.ones(M) / M
# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
#
# fig, axes = plt.subplots(3, 1, figsize=(21, 9), facecolor='white')
# titles = ['GW Barycenter (KDE)', 'OT Barycenter Space', 'Linear Interpolation']
#
# # --- GW INTERPOLATION ---
# C1 = ot.dist(X, X)
# C2 = ot.dist(Y, Y)
#
# for i, t in enumerate(ts):
#     if t == 0:
#         Z = X
#         sizes = a * scale
#     elif t == 1:
#         Z = Y
#         sizes = b * scale
#     else:
#         Cb = ot.gromov.gromov_barycenters(
#             N=M, Cs=[C1, C2], ps=[a, b], p=p,
#             lambdas=[1 - t, t], loss_fun='square_loss',
#             symmetric=True, max_iter=1000, tol=1e-9
#         )
#         sizes = 3 * np.ones(M)
#         Z = mds.fit_transform(Cb, X if t < 0.5 else Y)
#         Z = Z - Z.mean(0)[np.newaxis, :]
#         Z = Z / 10
#
#     kde = gaussian_kde(Z.T, weights=sizes / sizes.sum(), bw_method=0.11)
#     xgrid = np.linspace(-15, 15, 800)
#     ygrid = np.linspace(-15, 15, 800)
#     Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
#     positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
#     Z_kde = kde(positions).reshape(Xgrid.shape)
#     Z_kde = Z_kde / Z_kde.max()
#     Z_kde = Z_kde**0.5
#
#     x_shift = (1 - t) * -5+ t * 5
#     axes[0].imshow(Z_kde, extent=[-1 + x_shift, 1 + x_shift, -1, 1],
#                    origin='lower', cmap='gray_r', vmin=0, vmax=0.6, aspect='equal')
#     axes[0].text(x_shift, -1, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)
#
# axes[0].set_xlim(-5.8 , 5.8)
# axes[0].set_ylim(-1.5, 1.2)
#
# # --- OT INTERPOLATION ---
# C = ot.dist(X, Y)
# Gamma = ot.emd(a, b, C)
# GAMMA = Gamma.reshape(-1)
# XX = np.stack([X]*Y.shape[0], axis=1).reshape(-1, 2)
# DXX = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]).reshape(-1, 2)
#
# for i, t in enumerate(ts):
#     Xhat = XX - t * DXX
#     shifted_x = Xhat[:, 0] + (1 - t) * (-offset) + t * offset
#     axes[1].scatter(shifted_x, Xhat[:, 1], s=GAMMA * scale, c='k', alpha=0.8)
#     axes[1].text(np.mean(shifted_x), -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)
#
# axes[1].set_ylim(-20, 15)
#
# # --- EUCLIDEAN INTERPOLATION ---
# n = len(a)
# m = len(b)
# num_samples = 300
#
# for i, t in enumerate(ts):
#     if t == 0:
#         samples = X
#         sizes = a * scale
#     elif t == 1:
#         samples = Y
#         sizes = b * scale
#     else:
#         samples = []
#         sizes = []
#         for _ in range(num_samples):
#             if np.random.rand() < t:
#                 idx = np.random.choice(m, p=b)
#                 samples.append(Y[idx])
#                 weight = t * b[idx]
#             else:
#                 idx = np.random.choice(n, p=a)
#                 samples.append(X[idx])
#                 weight = (1 - t) * a[idx]
#             sizes.append(weight * scale)
#         samples = np.array(samples)
#         sizes = np.array(sizes)
#
#     shifted_x = samples[:, 0] + (1 - t) * (-offset) + t * offset
#     axes[2].scatter(shifted_x, samples[:, 1], s=sizes, alpha=0.8, c='k')
#     axes[2].text(np.mean(shifted_x), -17, f'({1 - t:.2f}, {t:.2f})', ha='center', va='top', fontsize=14)
#
# axes[2].set_ylim(-20, 15)
#
# # Final layout
# for i, ax in enumerate(axes):
#     ax.set_title(titles[i], fontsize=fs)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal')
#
# plt.tight_layout()
# plt.savefig("interpolation_all_combined_fixed.pdf", bbox_inches='tight', facecolor='white')
# plt.show()
