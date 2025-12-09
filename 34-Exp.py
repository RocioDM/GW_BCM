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
    diff = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def normalize_to_unit_square(X):
    """
    Affine-normalize a 2D point cloud into [0,1] x [0,1] for visualization.
    """
    X = X - X.min(axis=0)
    max_range = X.max(axis=0)
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
    Build an adjacency-based distance matrix for a generic CLOSED contour:
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
    """
    Geodesic distance along a CLOSED contour (loop):
      - adjacency between neighbors only,
      - shortest-path distance on that graph.
    """
    A = contour_adj_matrix(P)
    W = A.copy()
    n = len(W)

    mask_off = ~np.eye(n, dtype=bool)
    W[mask_off & (W == 0)] = np.inf
    np.fill_diagonal(W, 0.0)

    G = csr_matrix(W)
    return shortest_path(G, directed=False, method="D")

# =============================================================================
# Global noise levels and lambda trackers
# =============================================================================

# Common noise levels for all experiments
noise_levels = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25]

# To store the first lambda coordinate (associated with template 1 = circle)
lambda1_circle_geo = []
lambda1_circle_euc = []
lambda1_square_geo = []
lambda1_square_euc = []


###################################################################################################
## CIRCLE TARGET
###################################################################################################

###################################################################################################
## GEODESIC DISTANCE AS INTERNAL METRIC – CIRCLE TARGET WITH NOISE
###################################################################################################

print("=== GEODESIC INTERNAL METRIC — CIRCLE TARGET ===")
print("Building contour templates (circle & square)")

# Template 1: Circle with intrinsic arc-cos distance (only for neighbors)
n_circle = 80
circle_points_unit, circle_points_vis = build_circle_contour(n_points=n_circle, radius=1.0)
dist_circle_geo = circle_intrinsic_adj_matrix(circle_points_unit)
measure_circle = np.ones(n_circle) / float(n_circle)

# Template 2: Square with geodesic distance along its boundary
n_square = 80
square_points = build_square_contour(n_points=n_square)
square_points_vis = square_points.copy()  # already in [0,1]^2
dist_square_geo = geodesic_distance_matrix(square_points)
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle_geo, dist_square_geo]
measure_temp_list = [measure_circle, measure_square]

# Target: circle (same geometry as template 1), with noise added in a loop
M = n_circle
base_points_target = circle_points_unit.copy()
b = np.ones(M) / float(M)

rng = np.random.RandomState(0)

for noise_std in noise_levels:
    print("\n--- Geodesic metric (circle target): noise_std = {:.3f} ---".format(noise_std))

    # Add noise to the circle target (if noise_std = 0, it's just the clean circle)
    points_target = base_points_target + noise_std * rng.randn(M, 2)
    points_target_vis = normalize_to_unit_square(points_target)

    # Geodesic distance on the noisy circle
    B = geodesic_distance_matrix(points_target)

    print("Estimating lambda vector with our method and reconstructing barycenter")
    ## FP
    # B_recon, lambdas_est = utils.get_lambdas_constraints(
    #     matrix_temp_list, measure_temp_list, B, b
    # )

    ## BU
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    B_recon, lambdas_est = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)

    print(f"Lambdas coordinates (circle target, geodesic, noise {noise_std}): {lambdas_est}")
    lambda1_circle_geo.append(lambdas_est[0])   # store first coordinate (weight of circle template)

    B_recon = 0.5 * (B_recon + B_recon.T)
    np.fill_diagonal(B_recon, 0.0)

    # MDS embeddings
    print("Computing MDS embeddings (geodesic)")

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    points_B = mds.fit_transform(B)         # embedding of the (noisy) target metric
    points_B_recon = mds.fit_transform(B_recon)

    # Plot target vs reconstruction
    print("Plotting circle target vs reconstructed (geodesic)")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
    axes[0].set_title(f"Target circle (noise={noise_std})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
    axes[1].set_title("Target via MDS (geodesic)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
    axes[2].set_title("Reconstruction = Analysis+Synthesis+MDS")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


###################################################################################################
## AMBIENT SPACE DISTANCE AS INTERNAL METRIC – CIRCLE TARGET WITH NOISE
###################################################################################################

print("\n\n=== EUCLIDEAN (AMBIENT) INTERNAL METRIC — CIRCLE TARGET ===")
print("Building contour templates (circle & square)")

# Template 1: Circle with Euclidean distance in R^2
dist_circle_euc = euclidean_distance_matrix(circle_points_unit)
measure_circle = np.ones(n_circle) / float(n_circle)

# Template 2: Square with Euclidean distance in R^2
dist_square_euc = euclidean_distance_matrix(square_points)
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle_euc, dist_square_euc]
measure_temp_list = [measure_circle, measure_square]

# Target: circle (same geometry as template 1), with noise added in a loop
M = n_circle
base_points_target = circle_points_unit.copy()
b = np.ones(M) / float(M)

rng = np.random.RandomState(0)

for noise_std in noise_levels:
    print("\n--- Euclidean metric (circle target): noise_std = {:.3f} ---".format(noise_std))

    # Add noise to the circle target
    points_target = base_points_target + noise_std * rng.randn(M, 2)
    points_target_vis = normalize_to_unit_square(points_target)

    # Euclidean distance on the noisy circle
    B = euclidean_distance_matrix(points_target)

    print("Estimating lambda vector with our method and reconstructing barycenter")
    ## FP
    # B_recon, lambdas_est = utils.get_lambdas_constraints(
    #     matrix_temp_list, measure_temp_list, B, b
    # )

    ##BU
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    B_recon, lambdas_est = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)

    print(f"Lambdas coordinates (circle target, Euclidean, noise {noise_std}): {lambdas_est}")
    lambda1_circle_euc.append(lambdas_est[0])   # store first coordinate

    B_recon = 0.5 * (B_recon + B_recon.T)
    np.fill_diagonal(B_recon, 0.0)

    # MDS embeddings
    print("Computing MDS embeddings (Euclidean)")

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    points_B = mds.fit_transform(B)         # embedding of the (noisy) target metric
    points_B_recon = mds.fit_transform(B_recon)

    # Plot target vs reconstruction
    print("Plotting circle target vs reconstructed (Euclidean)")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
    axes[0].set_title(f"Target circle (noise={noise_std})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
    axes[1].set_title("Target via MDS (Euclidean)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
    axes[2].set_title("Reconstruction = Analysis+Synthesis+MDS")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


###################################################################################################
## SQUARE TARGET
###################################################################################################

###################################################################################################
## SQUARE AS TARGET — GEODESIC INTERNAL METRIC
###################################################################################################

print("\n\n===============================================")
print("=== GEODESIC INTERNAL METRIC — SQUARE TARGET ===")
print("===============================================")

# Templates: circle + square with geodesic internal metrics
dist_circle_geo = circle_intrinsic_adj_matrix(circle_points_unit)
measure_circle = np.ones(n_circle) / float(n_circle)

dist_square_geo = geodesic_distance_matrix(square_points)
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle_geo, dist_square_geo]
measure_temp_list = [measure_circle, measure_square]

# Target: square (same as template 2)
base_points_target = square_points.copy()
M = base_points_target.shape[0]
b = np.ones(M) / float(M)

rng = np.random.RandomState(1)   # different seed from circle

for noise_std in noise_levels:
    print(f"\n--- Square target (geodesic metric): noise_std = {noise_std} ---")

    # Noisy square
    points_target = base_points_target + noise_std * rng.randn(M, 2)
    points_target_vis = normalize_to_unit_square(points_target)

    B = geodesic_distance_matrix(points_target)

    # Decomposition and reconstruction

    ## FP
    # B_recon, lambdas_est = utils.get_lambdas_constraints(
    #     matrix_temp_list, measure_temp_list, B, b
    # )

    ##BU
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    B_recon, lambdas_est = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)

    print(f"Lambdas (square target, geodesic, noise {noise_std}): {lambdas_est}")
    lambda1_square_geo.append(lambdas_est[0])

    B_recon = 0.5 * (B_recon + B_recon.T)
    np.fill_diagonal(B_recon, 0.0)

    # MDS embeddings
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    points_B = mds.fit_transform(B)
    points_B_recon = mds.fit_transform(B_recon)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
    axes[0].set_title(f"Square target (noise={noise_std})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
    axes[1].set_title("Square target MDS (geodesic)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
    axes[2].set_title("Reconstruction  = Analysis+Synthesis+MDS")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


###################################################################################################
## SQUARE AS TARGET — EUCLIDEAN INTERNAL METRIC
###################################################################################################

print("\n\n=================================================")
print("=== EUCLIDEAN INTERNAL METRIC — SQUARE TARGET ===")
print("=================================================")

# Templates: circle + square (Euclidean versions)
n_circle = 80
circle_points_unit2, circle_points_vis2 = build_circle_contour(n_points=n_circle, radius=1.0)
dist_circle_E = euclidean_distance_matrix(circle_points_unit2)
measure_circle = np.ones(n_circle) / float(n_circle)

n_square = 80
square_points2 = build_square_contour(n_points=n_square)
square_points_vis2 = square_points2.copy()
dist_square_E = euclidean_distance_matrix(square_points2)
measure_square = np.ones(n_square) / float(n_square)

matrix_temp_list = [dist_circle_E, dist_square_E]
measure_temp_list = [measure_circle, measure_square]

# Target: square
base_points_target = square_points2.copy()
M = base_points_target.shape[0]
b = np.ones(M) / float(M)

rng = np.random.RandomState(2)

for noise_std in noise_levels:
    print(f"\n--- Square target (Euclidean metric): noise_std = {noise_std} ---")

    # Noisy square (extrinsic)
    points_target = base_points_target + noise_std * rng.randn(M, 2)
    points_target_vis = normalize_to_unit_square(points_target)

    B = euclidean_distance_matrix(points_target)

    # Reconstruction
    ## FP
    # B_recon, lambdas_est = utils.get_lambdas_constraints(
    #     matrix_temp_list, measure_temp_list, B, b
    # )

    ## BU
    B_bu, b_bu, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)
    B_recon, lambdas_est = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)

    print(f"Lambdas (square target, Euclidean, noise {noise_std}): {lambdas_est}")
    lambda1_square_euc.append(lambdas_est[0])

    B_recon = 0.5 * (B_recon + B_recon.T)
    np.fill_diagonal(B_recon, 0.0)

    # MDS embeddings
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    points_B = mds.fit_transform(B)
    points_B_recon = mds.fit_transform(B_recon)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(points_target_vis[:, 0], points_target_vis[:, 1], s=b * 350)
    axes[0].set_title(f"Square target (noise={noise_std})")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(points_B[:, 0], points_B[:, 1], s=b * 350)
    axes[1].set_title("Square target MDS (Euclidean)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].scatter(points_B_recon[:, 0], points_B_recon[:, 1], s=b * 350)
    axes[2].set_title("Reconstruction = Analysis+Synthesis+MDS")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()


###################################################################################################
## SUMMARY: PLOT LAMBDA_1 VS NOISE FOR ALL 4 CASES
###################################################################################################

plt.figure(figsize=(8, 6))

plt.plot(noise_levels, lambda1_circle_geo, marker='o', label='Circle target – Geodesic')
plt.plot(noise_levels, lambda1_circle_euc, marker='s', label='Circle target – Euclidean')
# plt.plot(noise_levels, lambda1_square_geo, marker='^', label='Square target – Geodesic')
# plt.plot(noise_levels, lambda1_square_euc, marker='x', label='Square target – Euclidean')

plt.xlabel('Noise standard deviation')
plt.ylabel('First lambda coordinate (template 1 = circle)')
plt.title('Behavior of λ₁ vs noise level across noise perturbations of circle template')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

# plt.plot(noise_levels, lambda1_circle_geo, marker='o', label='Circle target – Geodesic')
# plt.plot(noise_levels, lambda1_circle_euc, marker='s', label='Circle target – Euclidean')
#plt.plot(noise_levels, lambda1_square_geo, marker='^', label='Square target – Geodesic')
plt.plot(noise_levels, lambda1_square_euc, marker='x', label='Square target – Euclidean')

plt.xlabel('Noise standard deviation')
plt.ylabel('First lambda coordinate (template 1 = circle)')
plt.title('Behavior of λ₁ vs noise')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
