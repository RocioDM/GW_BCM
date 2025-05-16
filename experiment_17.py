import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.gromov

# Step 1: Generate two point clouds
n = 30
t = np.linspace(0, 2 * np.pi, n)
X = np.stack((np.cos(t), np.sin(t)), axis=1)  # circle
Y = np.stack((1.5 * np.cos(t + 0.5), 0.5 * np.sin(t + 0.5)), axis=1)  # ellipse, rotated

# Step 2: Uniform distributions
p = ot.unif(n)
q = ot.unif(n)


# Or non-uniform point clouds
n_x = 30
n_y = 50

t_x = np.linspace(0, 2 * np.pi, n_x)
t_y = np.linspace(0, 2 * np.pi, n_y)

X = np.stack((np.cos(t_x), np.sin(t_x)), axis=1)
Y = np.stack((1.5 * np.cos(t_y + 0.5), 0.5 * np.sin(t_y + 0.5)), axis=1)

# Non-uniform size distributions
p = ot.unif(n_x)  # size 30
q = ot.unif(n_y)  # size 50


import utils

## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()

## Select two point clouds of digits 0 and 1

ind1=10
ind2=5

p = Data[ind1,:,2]
X = Data[ind1,p!=-1,:2]
X = X-X.mean(0)[np.newaxis,:]
# X += np.random.randn(X.shape[0],X.shape[1])*.1
X[:,0]-= 10

q = Data[ind2,:,2]
Y = Data[ind2,q!=-1,:2]
Y = Y-Y.mean(0)[np.newaxis,:]
# Y += np.random.randn(Y.shape[0],Y.shape[1])*.1
Y[:,0]+= 10


n_x = len(X)
n_y = len(Y)
# n = min(len(X), len(Y))

X = X[:n_x]
Y = Y[:n_y]
p = p[:n_x]
q = q[:n_y]

p = p[p!=-1]
q= q[q!=-1]

p = p/float(p.sum())
q = q/float(q.sum())








# Step 3: Compute cost matrices
C1 = ot.dist(X, X)
C2 = ot.dist(Y, Y)
M = ot.dist(X, Y)

# Step 4a: Classical OT
T_ot = ot.emd(p, q, M)

# Step 4b: Gromov-Wasserstein
T_gw = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss')




# Step 5: Visualization
def plot_transport(X, Y, T, title):
    n_x, n_y = T.shape  # dynamically get the sizes from the transport matrix
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], label='X', c='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label='Y', c='red')
    max_T = T.max()
    for i in range(n_x):
        for j in range(n_y):
            if T[i, j] > 1e-4:  # lower threshold
                alpha = T[i, j] / max_T
                plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'gray', alpha=alpha/10)
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.show()


# def plot_transport(X, Y, T, title):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(X[:, 0], X[:, 1], label='X', c='blue')
#     plt.scatter(Y[:, 0], Y[:, 1], label='Y', c='red')
#     for i in range(n):
#         for j in range(n):
#             if T[i, j] > 1e-2:
#                 plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k-', alpha=T[i, j]*5)
#     plt.legend()
#     plt.title(title)
#     plt.axis('equal')
#     plt.show()

plot_transport(X, Y, T_ot, 'Classical OT')
plot_transport(X, Y, T_gw, 'Gromov-Wasserstein OT')

plt.figure(figsize=(6, 5))
plt.imshow(T_ot, cmap='viridis')
plt.colorbar(label='Transport Mass')
plt.title('Classical OT Transport Plan')
plt.xlabel('Target Index (Y)')
plt.ylabel('Source Index (X)')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(T_gw, cmap='viridis')
plt.colorbar(label='Transport Mass')
plt.title('GW Transport Plan')
plt.xlabel('Target Index (Y)')
plt.ylabel('Source Index (X)')
plt.show()


