import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Helper: align embeddings (orthogonal Procrustes)
# -----------------------------
def align_to_reference(X, Xref):
    """
    Align X to Xref by optimal rotation/reflection + translation (least squares).
    Keeps scales as-is (so points don't shrink/grow across frames).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Rc = Xref - Xref.mean(axis=0, keepdims=True)

    # solve: min_R || Xc R - Rc ||, with R orthogonal
    U, _, Vt = np.linalg.svd(Xc.T @ Rc)
    R = U @ Vt
    return Xc @ R + Xref.mean(axis=0, keepdims=True)

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



# -----------------------------
# Movie parameters
# -----------------------------
p_list = [1, 1.5, 2, 4, 10]
gif_path = "pwspd_recon_p_sweep.gif"

# Your ellipse points + measure (reuse from your code if already defined)
M = 120
theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
x = 1.0 * np.cos(theta)
y = 0.6 * np.sin(theta)
points_target = np.stack([x, y], axis=1)
b = np.ones(M) / float(M)

# Templates point sets (fixed geometry; only the PWSPD metric changes with p)
n_circle = 100
circle_points_unit, circle_points_vis = build_circle_contour(n_points=n_circle, radius=1.0)
measure_circle = np.ones(n_circle) / float(n_circle)

n_square = 100
square_points = build_square_contour(n_points=n_square)
square_points_vis = square_points.copy()
measure_square = np.ones(n_square) / float(n_square)

# We'll store embeddings + metadata for all p
embeddings = []
lambdas_all = []
gw_all = []

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)

# -----------------------------
# Compute all frames
# -----------------------------
for p_pwspd in p_list:
    print(f"Computing frame for p={p_pwspd} ...")

    # PWSPD distance matrices for templates
    dist_circle = pwspd_distance_matrix(circle_points_unit, p=p_pwspd)
    dist_square = pwspd_distance_matrix(square_points, p=p_pwspd)

    matrix_temp_list = [dist_circle, dist_square]
    measure_temp_list = [measure_circle, measure_square]

    # PWSPD distance matrix for target ellipse
    B = pwspd_distance_matrix(points_target, p=p_pwspd)

    # Reconstruction + lambdas
    B_recon, lambdas_est = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)
    B_recon = 0.5 * (B_recon + B_recon.T)
    np.fill_diagonal(B_recon, 0.0)

    # Optional: GW distance (can be slow; comment out if you want)
    gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
    gw_dist = gromov_distance["gw_dist"]

    # MDS embedding of the reconstruction
    points_B_recon = mds.fit_transform(B_recon)

    embeddings.append(points_B_recon)
    lambdas_all.append(lambdas_est)
    gw_all.append(gw_dist)

# Align all embeddings to the reference (pick p=2 if present, else the first)
if 2 in p_list:
    ref_idx = p_list.index(2)
else:
    ref_idx = 0

Xref = embeddings[ref_idx]
embeddings_aligned = []
for X in embeddings:
    embeddings_aligned.append(align_to_reference(X, Xref))

# Set consistent axis limits across frames
all_pts = np.vstack(embeddings_aligned)
pad = 0.05 * max(all_pts[:, 0].ptp(), all_pts[:, 1].ptp())
xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

# -----------------------------
# Build the animation
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 6))

sc = ax.scatter([], [], s=b * 350)
title = ax.set_title("")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

def init():
    sc.set_offsets(np.zeros((M, 2)))
    title.set_text("")
    return sc, title

def update(frame_idx):
    X = embeddings_aligned[frame_idx]
    sc.set_offsets(X)

    p_val = p_list[frame_idx]
    lam = lambdas_all[frame_idx]
    gw = gw_all[frame_idx]
    title.set_text(f"Reconstruction MDS (PWSPD) | p={p_val} | GW={gw:.4g} | lambdas={np.array(lam)}")
    return sc, title

anim = FuncAnimation(fig, update, frames=len(p_list), init_func=init, blit=True)

# Save as GIF (no ffmpeg needed)
anim.save(gif_path, writer=PillowWriter(fps=1))  # fps=1 => 1 frame per second
plt.close(fig)

print(f"Saved GIF to: {gif_path}")
