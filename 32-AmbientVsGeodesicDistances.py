import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import ot
import utils

# =============================================================================
# Helper functions
# =============================================================================

def euclidean_distance_matrix(P):
    diff = P[:,None,:] - P[None,:,:]
    return np.linalg.norm(diff, axis=-1)

def normalize_to_unit_square(X):
    """
    Affine-normalize a 2D point cloud into [0,1] x [0,1] for visualization.
    """
    X = X - X.min(axis=0)
    max_range = X.max(axis=0)
    # Avoid division by zero if degenerate
    max_range[max_range == 0] = 1.0
    X = X / max_range
    return X


def build_circle_contour(n_points=100, radius=1.0):
    """
    Build a discretized circle in R^2:
      - Returns:
          points_unit: points on the unit circle (for intrinsic distance)
          points_vis: same points normalized into [0,1]^2 for plotting
    """
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points_unit = np.stack([np.cos(thetas), np.sin(thetas)], axis=1) * radius
    points_vis = normalize_to_unit_square(points_unit)
    return points_unit, points_vis


def circle_intrinsic_adj_matrix(points_unit):
    """
    Build an adjacency-based distance matrix for a circle:
      - Non-zero only between consecutive points (including last<->first),
      - Distance between neighbors is arccos(<x_i, x_j>) on the unit circle.
    Assumes points_unit lie on (or very close to) the unit circle.
    """
    n = points_unit.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = (i + 1) % n  # neighbor (wrap-around to enforce closed contour)
        # dot product for unit vectors
        dot = np.clip(np.dot(points_unit[i], points_unit[j]), -1.0, 1.0)
        dist = np.arccos(dot)
        D[i, j] = dist
        D[j, i] = dist

    return D


def build_square_contour(n_points=100):
    """
    Build a discretized square contour on [0,1]^2:
      - Points ordered around the square (closed contour).
      - Returns points_vis in [0,1]^2.
    """
    # Ensure n_points is divisible by 4 for equal sampling on each side
    n_side = n_points // 4
    n_points = n_side * 4

    # Bottom side: (0,0) -> (1,0)
    t = np.linspace(0, 1, n_side, endpoint=False)
    bottom = np.stack([t, np.zeros_like(t)], axis=1)

    # Right side: (1,0) -> (1,1)
    t = np.linspace(0, 1, n_side, endpoint=False)
    right = np.stack([np.ones_like(t), t], axis=1)

    # Top side: (1,1) -> (0,1)
    t = np.linspace(1, 0, n_side, endpoint=False)
    top = np.stack([t, np.ones_like(t)], axis=1)

    # Left side: (0,1) -> (0,0)
    t = np.linspace(1, 0, n_side, endpoint=False)
    left = np.stack([np.zeros_like(t), t], axis=1)

    points = np.concatenate([bottom, right, top, left], axis=0)
    # already in [0,1]^2
    return points


def contour_adj_matrix(points):
    """
    Build an adjacency-based distance matrix for a generic contour:
      - Non-zero only between consecutive points (including last<->first),
      - Distance is Euclidean between neighbors.
    """
    n = points.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = (i + 1) % n  # neighbor (wrap-around)
        dist = np.linalg.norm(points[i] - points[j])
        D[i, j] = dist
        D[j, i] = dist

    return D


def geodesic_distance_matrix(P):
    A = contour_adj_matrix(P)
    W = A.copy()
    n = len(W)

    mask_off = ~np.eye(n, dtype=bool)
    W[mask_off & (W == 0)] = np.inf
    np.fill_diagonal(W, 0.0)

    G = csr_matrix(W)
    return shortest_path(G, directed=False, method="D")


def pwspd_distance_matrix(P, p=2, method="FW"):
    """
    Discrete p-weighted shortest path distance (PWSPD) from Def. 1.1 of arXiv:2012.09385.

    For a point set X = {x_i}, define edge weights w_ij = ||x_i - x_j||^p (complete graph).
    Then ℓ_p(i,j) = ( shortest_path_sum_w(i,j) )^(1/p).

    Parameters
    ----------
    P : (n, d) array
        Point cloud.
    p : float >= 1
        Power parameter (default 2).
    method : str
        shortest_path method. "FW" is Floyd–Warshall (dense, O(n^3)).

    Returns
    -------
    L : (n, n) array
        PWSPD distance matrix.
    """
    D = euclidean_distance_matrix(P)
    W = D ** p
    np.fill_diagonal(W, 0.0)

    # Since this is the complete graph, all off-diagonal weights are present.
    # shortest_path returns minimal summed weights; take (.)^(1/p) after.
    sp = shortest_path(W, directed=False, method=method)
    return sp ** (1.0 / p)


###################################################################################################
## GEODESIC DISTANCE AS INTERNAL METRIC ###########################################################
###################################################################################################


# =============================================================================
# BUILD TWO TEMPLATES: CIRCLE & SQUARE
# =============================================================================

print("Building contour templates (circle & square)")

# Template 1: Circle with intrinsic arc-cos distance (only for neighbors)
n_circle = 100
circle_points_unit, circle_points_vis = build_circle_contour(n_points=n_circle, radius=1.0)
dist_circle = circle_intrinsic_adj_matrix(circle_points_unit)
#print(np.shape(dist_circle))
measure_circle = np.ones(n_circle) / float(n_circle)

# Template 2: Square with neighbor-only Euclidean distance
n_square = 100
square_points = build_square_contour(n_points=n_square)
square_points_vis = square_points.copy()  # already in [0,1]^2
dist_square = geodesic_distance_matrix(square_points)
#print(np.shape(dist_square))
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle, dist_square]
measure_temp_list = [measure_circle, measure_square]
n_temp = len(matrix_temp_list)

# =============================================================================
# GENERATE INPUT
# =============================================================================

print("Generating an input")


# -------------------------------------------------------------------------
# TARGET: B comes from the contour of a 2D shape
# -------------------------------------------------------------------------

# EXAMPLE 1: use a parametric contour (e.g., ellipse/star/etc.)
M = 120  # number of contour points (and matrix size)

theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
# Example: slightly anisotropic ellipse just so it's not a circle
x = 1.0 * np.cos(theta)
y = 0.6 * np.sin(theta)
points_target = np.stack([x, y], axis=1)        # shape (M, 2)
points_target_vis = normalize_to_unit_square(points_target)

# Build distance matrix: non-zero ONLY for consecutive points
B = geodesic_distance_matrix(points_target)
#print(np.shape(B))
# Probability measure supported on the contour points
b = np.ones(M) / float(M)


###################################################################################################
# EXAMPLE 2: Ellipse contour with noise
###################################################################################################

# Add noise to the contour
noise_std = 0.1  # you can tune this
rng = np.random.RandomState(0)
points_target_noisy = points_target + noise_std * rng.randn(M, 2)

# Normalize to [0,1]^2 just for visualization
points_target_vis_noisy = normalize_to_unit_square(points_target_noisy)

# Build distance matrix: non-zero ONLY for consecutive points
B_noisy = geodesic_distance_matrix(points_target_noisy)
#print(np.shape(B_noisy))

# Probability measure supported on the contour points
b_noisy = np.ones(M) / float(M)



###################################################################################################
print("Estimating lambda vector with our method and reconstructing barycenter")
###################################################################################################

###################################################################################################
## Example 1 ######################################################################################
###################################################################################################

## FP
B_recon, lambdas_est = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)

print(f'Lambdas coordinates {lambdas_est}')

B_recon = 0.5 * (B_recon + B_recon.T)
np.fill_diagonal(B_recon, 0.0)

# GW-distance between Original Input B and Reconstructed B_recon
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance["gw_dist"]
print(f"[Example 1] GW(Target, Reconstructed Target): {gw_dist}")


###################################################################################################
## Example 2 ######################################################################################
###################################################################################################

## FP
B_recon_noisy, lambdas_est_noisy = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B_noisy, b_noisy)


print(f'Lambdas coordinates for noisy input {lambdas_est_noisy}')

B_recon_noisy = 0.5 * (B_recon_noisy + B_recon_noisy.T)
np.fill_diagonal(B_recon_noisy, 0.0)

# GW-distance between Original Input B and Reconstructed B_recon_noisy
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon_noisy, b, b_noisy, log=True)[1]
gw_dist = gromov_distance["gw_dist"]
print(f"[Example 2] GW(Target (clean ellipse), Reconstruction from noisy ellipse): {gw_dist}")



# =============================================================================
# MDS EMBEDDINGS
# =============================================================================

print("Computing MDS embeddings")

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)

# Example 1
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

#points_B = normalize_to_unit_square(points_B)
#points_B_recon = normalize_to_unit_square(points_B_recon)

# Example 2
points_B_recon_noisy = mds.fit_transform(B_recon_noisy)
#points_B_recon_noisy = normalize_to_unit_square(points_B_recon_noisy)


# =============================================================================
# PLOT TEMPLATES (CIRCLE & SQUARE)
# =============================================================================

print("Plotting templates")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

# Circle template
axes[0].scatter(circle_points_vis[:, 0], circle_points_vis[:, 1],
                s=measure_circle * 250)
axes[0].set_title("Template 1: Circle contour")
axes[0].set_aspect("equal", adjustable="box")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Square template
axes[1].scatter(square_points_vis[:, 0], square_points_vis[:, 1],
                s=measure_square * 250)
axes[1].set_title("Template 2: Square contour")
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# =============================================================================
# PLOT B AND B_RECON (ELLIPSE)
# =============================================================================

print("Plotting input vs reconstructed (ellipse)")

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
axes[0].set_title("Input (ellipse)")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
axes[1].set_title("Input via MDS (ellipse)")
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
axes[2].set_title("Reconstruction = Analysis+Synthesis+MDS")
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.show()

print("Plotting ellipse: input with noise vs reconstructed")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# clean contour
axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1],
                s=b * 350, label="clean")
for i in range(M):
    j = (i + 1) % M
    axes[0].plot(
        [points_target_vis[i, 0], points_target_vis[j, 0]],
        [points_target_vis[i, 1], points_target_vis[j, 1]],
        linewidth=0.5
    )
# noisy contour
axes[0].scatter(points_target_vis_noisy[:, 0], points_target_vis_noisy[:, 1],
                s=b * 150, alpha=0.7, label="noisy")
axes[0].set_title("Elliptic contour: clean vs noisy (no MDS)")
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend()

axes[1].scatter(points_B_recon_noisy[:, 0], points_B_recon_noisy[:, 1], s=b * 350)
axes[1].set_title("Reconstruction - Noisy Ellipse Input")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()



###################################################################################################
###################################################################################################


###################################################################################################
## AMBIENT SPACE DISTANCE AS INTERNAL METRIC ######################################################
###################################################################################################


# =============================================================================
# BUILD TWO TEMPLATES: CIRCLE & SQUARE
# =============================================================================

print("Building contour templates (circle & square)")

# Template 1: Circle with intrinsic arc-cos distance (only for neighbors)
n_circle = 100
circle_points_unit, circle_points_vis = build_circle_contour(n_points=n_circle, radius=1.0)
dist_circle = euclidean_distance_matrix(circle_points_unit)
measure_circle = np.ones(n_circle) / float(n_circle)

# Template 2: Square with neighbor-only Euclidean distance
n_square = 100
square_points = build_square_contour(n_points=n_square)
square_points_vis = square_points.copy()  # already in [0,1]^2
dist_square= euclidean_distance_matrix(square_points)
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle, dist_square]
measure_temp_list = [measure_circle, measure_square]
n_temp = len(matrix_temp_list)

# =============================================================================
# GENERATE INPUT
# =============================================================================

print("Generating an input")


# -------------------------------------------------------------------------
# TARGET: B comes from the contour of a 2D shape
# -------------------------------------------------------------------------

# EXAMPLE 1: use a parametric contour (e.g., ellipse/star/etc.)
M = 120  # number of contour points (and matrix size)

theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
# Example: slightly anisotropic ellipse just so it's not a circle
x = 1.0 * np.cos(theta)
y = 0.6 * np.sin(theta)
points_target = np.stack([x, y], axis=1)        # shape (M, 2)
points_target_vis = normalize_to_unit_square(points_target)


B = euclidean_distance_matrix(points_target)
# Probability measure supported on the contour points
b = np.ones(M) / float(M)


###################################################################################################
# EXAMPLE 2: Ellipse contour with noise
###################################################################################################

# Add noise to the contour
noise_std = 0.1
rng = np.random.RandomState(0)
points_target_noisy = points_target + noise_std * rng.randn(M, 2)

# Normalize to [0,1]^2 just for visualization
points_target_vis_noisy = normalize_to_unit_square(points_target_noisy)

B_noisy = euclidean_distance_matrix(points_target_noisy)

# Probability measure supported on the contour points
b_noisy = np.ones(M) / float(M)



###################################################################################################
print("Estimating lambda vector with our method and reconstructing barycenter")
###################################################################################################

###################################################################################################
## Example 1 ######################################################################################
###################################################################################################

# FP
B_recon, lambdas_est = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)

print(f'Lambdas coordinates {lambdas_est}')

B_recon = 0.5 * (B_recon + B_recon.T)
np.fill_diagonal(B_recon, 0.0)

# GW-distance between Original Input B and Reconstructed B_recon
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
gw_dist = gromov_distance["gw_dist"]
print(f"[Example 1] GW(Target, Reconstructed Target): {gw_dist}")


###################################################################################################
## Example 2 ######################################################################################
###################################################################################################

## FP
B_recon_noisy, lambdas_est_noisy = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B_noisy, b_noisy)


print(f'Lambdas coordinates for noisy input {lambdas_est_noisy}')

B_recon_noisy = 0.5 * (B_recon_noisy + B_recon_noisy.T)
np.fill_diagonal(B_recon_noisy, 0.0)

# GW-distance between Original Input B and Reconstructed B_recon_noisy
gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon_noisy, b, b_noisy, log=True)[1]
gw_dist = gromov_distance["gw_dist"]
print(f"[Example 2] GW(Target (clean ellipse), Reconstruction from noisy ellipse): {gw_dist}")



# =============================================================================
# MDS EMBEDDINGS
# =============================================================================

print("Computing MDS embeddings")

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)

# Example 1
points_B = mds.fit_transform(B)
points_B_recon = mds.fit_transform(B_recon)

#points_B = normalize_to_unit_square(points_B)
#points_B_recon = normalize_to_unit_square(points_B_recon)

# Example 2
points_B_recon_noisy = mds.fit_transform(B_recon_noisy)
#points_B_recon_noisy = normalize_to_unit_square(points_B_recon_noisy)


# =============================================================================
# PLOT TEMPLATES (CIRCLE & SQUARE)
# =============================================================================

print("Plotting templates")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

# Circle template
axes[0].scatter(circle_points_vis[:, 0], circle_points_vis[:, 1],
                s=measure_circle * 250)
axes[0].set_title("Template 1: Circle contour")
axes[0].set_aspect("equal", adjustable="box")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Square template
axes[1].scatter(square_points_vis[:, 0], square_points_vis[:, 1],
                s=measure_square * 250)
axes[1].set_title("Template 2: Square contour")
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# =============================================================================
# PLOT B AND B_RECON (ELLIPSE)
# =============================================================================

print("Plotting input vs reconstructed (ellipse)")

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
axes[0].set_title("Input (ellipse)")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
axes[1].set_title("Input via MDS (ellipse)")
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
axes[2].set_title("Reconstruction = Analysis+Synthesis+MDS")
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.show()

print("Plotting ellipse: input with noise vs reconstructed")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# clean contour
axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1],
                s=b * 350, label="clean")
for i in range(M):
    j = (i + 1) % M
    axes[0].plot(
        [points_target_vis[i, 0], points_target_vis[j, 0]],
        [points_target_vis[i, 1], points_target_vis[j, 1]],
        linewidth=0.5
    )
# noisy contour
axes[0].scatter(points_target_vis_noisy[:, 0], points_target_vis_noisy[:, 1],
                s=b * 150, alpha=0.7, label="noisy")
axes[0].set_title("Elliptic contour: clean vs noisy (no MDS)")
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend()

axes[1].scatter(points_B_recon_noisy[:, 0], points_B_recon_noisy[:, 1], s=b * 350)
axes[1].set_title("Reconstruction - Noisy Ellipse Input")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()


