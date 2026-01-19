import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

from sklearn.manifold import MDS
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import utils
import ot


# ============================================================
# 1) Utilities
# ============================================================

def align_to_reference(X: np.ndarray, Xref: np.ndarray) -> np.ndarray:
    """
    Align X to Xref by optimal rotation/reflection + translation (orthogonal Procrustes).
    Keeps scales as-is.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Rc = Xref - Xref.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Xc.T @ Rc)
    R = U @ Vt
    return Xc @ R + Xref.mean(axis=0, keepdims=True)


# ============================================================
# 2) Shape builders
# ============================================================

def build_circle(n: int = 100, radius: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1) * radius


def build_ellipse(n: int = 120, a: float = 1.0, b: float = 0.6) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def build_square_contour(n_points: int = 100) -> np.ndarray:
    """
    Square contour in [0,1]^2 sampled in order (cycle).
    """
    n_side = max(1, n_points // 4)
    n_points = 4 * n_side

    t = np.linspace(0, 1, n_side, endpoint=False)
    bottom = np.stack([t, np.zeros_like(t)], axis=1)

    t = np.linspace(0, 1, n_side, endpoint=False)
    right = np.stack([np.ones_like(t), t], axis=1)

    t = np.linspace(1, 0, n_side, endpoint=False)
    top = np.stack([t, np.ones_like(t)], axis=1)

    t = np.linspace(1, 0, n_side, endpoint=False)
    left = np.stack([np.zeros_like(t), t], axis=1)

    return np.concatenate([bottom, right, top, left], axis=0)


def build_superellipse(n: int = 120, m: float = 4.0, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """
    Parametric superellipse (Lamé curve):
      x = a * sign(cos t) * |cos t|^{2/m}
      y = b * sign(sin t) * |sin t|^{2/m}
    m=2 -> circle/ellipse, m->inf -> square-like.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * (np.abs(ct) ** (2.0 / m))
    y = b * np.sign(st) * (np.abs(st) ** (2.0 / m))
    return np.stack([x, y], axis=1)


# ============================================================
# 3) Distance backends (ambient / geodesic / p-Fermat)
# ============================================================

def ambient_distance_matrix(P: np.ndarray) -> np.ndarray:
    return cdist(P, P, metric="euclidean")


def _knn_graph_weights(P: np.ndarray, k: int, edge_weight: Callable[[float], float]) -> csr_matrix:
    """
    Build a symmetric kNN graph with weights w_ij = edge_weight(||P_i - P_j||).
    """
    D = cdist(P, P, metric="euclidean")
    n = D.shape[0]
    rows, cols, data = [], [], []

    for i in range(n):
        idx = np.argsort(D[i])[1:k + 1]
        for j in idx:
            w = edge_weight(D[i, j])
            rows.append(i)
            cols.append(j)
            data.append(w)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Make undirected (more connected than minimum)
    W = W.maximum(W.T)
    return W


def _cycle_graph_weights(P: np.ndarray, edge_weight: Callable[[float], float]) -> csr_matrix:
    """
    For ordered closed contours (points already in order around the curve),
    connect i to i+1 and i-1 (cycle graph).
    """
    D = cdist(P, P, metric="euclidean")
    n = D.shape[0]
    rows, cols, data = [], [], []

    for i in range(n):
        j1 = (i + 1) % n
        j2 = (i - 1) % n
        rows += [i, i]
        cols += [j1, j2]
        data += [edge_weight(D[i, j1]), edge_weight(D[i, j2])]

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    return W


def geodesic_distance_matrix(
    P: np.ndarray,
    graph: str = "knn",
    k: int = 10,
    assume_ordered_cycle: bool = False,
) -> np.ndarray:
    """
    Approximate geodesic distance by shortest path with Euclidean edge weights.
    """
    edge_w = lambda d: d

    if assume_ordered_cycle or graph == "cycle":
        W = _cycle_graph_weights(P, edge_w)
    else:
        W = _knn_graph_weights(P, k=k, edge_weight=edge_w)

    sp = shortest_path(W, directed=False, method="D")

    # Safety: if disconnected, replace inf by a large finite value (so POT doesn't crash)
    if not np.isfinite(sp).all():
        finite = sp[np.isfinite(sp)]
        big = 2.0 * finite.max() if finite.size else 1.0
        sp = np.where(np.isfinite(sp), sp, big)

    return sp


def fermat_distance_matrix(
    P: np.ndarray,
    p: float = 2.0,
    graph: str = "knn",
    k: int = 10,
    assume_ordered_cycle: bool = False,
) -> np.ndarray:
    """
    p-Fermat distance:
      edge weights w_ij = ||xi-xj||^p
      shortest path sum of w
      take (.)^(1/p)
    """
    edge_w = lambda d: d ** p

    if assume_ordered_cycle or graph == "cycle":
        W = _cycle_graph_weights(P, edge_w)
    else:
        W = _knn_graph_weights(P, k=k, edge_weight=edge_w)

    sp = shortest_path(W, directed=False, method="D")

    # Safety for disconnected graphs
    if not np.isfinite(sp).all():
        finite = sp[np.isfinite(sp)]
        big = 2.0 * finite.max() if finite.size else 1.0
        sp = np.where(np.isfinite(sp), sp, big)

    return sp ** (1.0 / p)


DistanceFn = Callable[..., np.ndarray]
DISTANCE_BUILDERS: Dict[str, DistanceFn] = {
    "ambient": lambda P, **kw: ambient_distance_matrix(P),
    "geodesic": lambda P, **kw: geodesic_distance_matrix(P, **kw),
    "fermat": lambda P, **kw: fermat_distance_matrix(P, **kw),
}


# ============================================================
# 4) Experiment pipeline
# ============================================================

@dataclass
class Shape:
    name: str
    points: np.ndarray
    measure: np.ndarray
    ordered_cycle: bool = True


def make_uniform_measure(n: int) -> np.ndarray:
    return np.ones(n) / float(n)


def build_distance(P: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    if mode not in DISTANCE_BUILDERS:
        raise ValueError(f"Unknown distance mode: {mode}. Choose from {list(DISTANCE_BUILDERS.keys())}")
    D = DISTANCE_BUILDERS[mode](P, **kwargs)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def run_reconstruction_sweep(
    target: Shape,
    templates: List[Shape],
    distance_mode: str,
    sweep_values: List[Optional[float]],
    sweep_param_name: str,
    distance_kwargs_base: Dict[str, Any],
    gif_path: str,
    fps: int = 1,
    mds_random_state: int = 42,
):
    """
    Sweep over one distance parameter (e.g. Fermat p, k for knn, etc.), or run a single frame.
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=mds_random_state)

    embeddings = []
    lambdas_all = []
    gw_all = []

    for val in sweep_values:
        if sweep_param_name:
            print(f"Frame: {sweep_param_name}={val}")
        else:
            print("Frame: single run")

        # Build kwargs for this frame; only inject sweep param if it's real
        distance_kwargs = dict(distance_kwargs_base)
        if sweep_param_name:
            distance_kwargs[sweep_param_name] = val

        def kwargs_for(shape: Shape):
            kw = dict(distance_kwargs)
            if distance_mode in ("geodesic", "fermat"):
                kw.setdefault("assume_ordered_cycle", shape.ordered_cycle)
            return kw

        # Distances for templates
        matrix_temp_list = [build_distance(s.points, distance_mode, **kwargs_for(s)) for s in templates]
        measure_temp_list = [s.measure for s in templates]

        # Distance for target
        B = build_distance(target.points, distance_mode, **kwargs_for(target))

        # Reconstruction + lambdas
        B_recon, lambdas_est = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, B, target.measure
        )
        B_recon = 0.5 * (B_recon + B_recon.T)
        np.fill_diagonal(B_recon, 0.0)

        # GW distance (optional but you’re using it)
        gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, target.measure, target.measure, log=True)[1]
        gw_dist = gromov_distance["gw_dist"]

        # MDS embedding
        X = mds.fit_transform(B_recon)

        embeddings.append(X)
        lambdas_all.append(lambdas_est)
        gw_all.append(gw_dist)

    # Align frames
    ref_idx = len(sweep_values) // 2
    Xref = embeddings[ref_idx]
    embeddings = [align_to_reference(X, Xref) for X in embeddings]

    # Consistent axis limits (NumPy 2.0 compatible)
    all_pts = np.vstack(embeddings)
    range_x = np.ptp(all_pts[:, 0])
    range_y = np.ptp(all_pts[:, 1])
    scale = max(range_x, range_y, 1e-12)
    pad = 0.05 * scale

    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

    # Animation: initialize with first frame so sizes match
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(embeddings[0][:, 0], embeddings[0][:, 1], s=target.measure * 350)
    title = ax.set_title("")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    def init():
        sc.set_offsets(embeddings[0])
        title.set_text("")
        return sc, title

    def update(i):
        sc.set_offsets(embeddings[i])
        lam = lambdas_all[i]
        gw = gw_all[i]

        if sweep_param_name:
            val = sweep_values[i]
            param_txt = f" | {sweep_param_name}={val}"
        else:
            param_txt = ""

        title.set_text(
            f"MDS(recon) | dist={distance_mode}{param_txt} | GW={gw:.4g} | lambdas={np.array(lam)}"
        )
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(sweep_values), init_func=init, blit=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


# ============================================================
# 5) Example usage
# ============================================================

n_temp = 120
circle = Shape("circle", build_circle(n_temp, radius=1.0), make_uniform_measure(n_temp), ordered_cycle=True)
square = Shape("square", build_square_contour(n_temp), make_uniform_measure(n_temp), ordered_cycle=True)
templates = [circle, square]

# Target (recommended): superellipse between circle and square
target_super = Shape(
    "superellipse_m=4",
    build_superellipse(120, m=4.0, a=1.0, b=1.0),
    make_uniform_measure(120),
    ordered_cycle=True,
)

# --- Fermat p-sweep ---
p_list = [1, 1.5, 2, 4, 10]
run_reconstruction_sweep(
    target=target_super,
    templates=templates,
    distance_mode="fermat",
    sweep_values=p_list,
    sweep_param_name="p",
    distance_kwargs_base={"graph": "cycle"},
    gif_path="recon_fermat_p_sweep.gif",
    fps=1,
)

# --- Geodesic (single run) ---
run_reconstruction_sweep(
    target=target_super,
    templates=templates,
    distance_mode="geodesic",
    sweep_values=[None],
    sweep_param_name="",           # <-- no dummy injected
    distance_kwargs_base={"graph": "cycle"},
    gif_path="recon_geodesic.gif",
    fps=1,
)

# --- Ambient (single run) ---
run_reconstruction_sweep(
    target=target_super,
    templates=templates,
    distance_mode="ambient",
    sweep_values=[None],
    sweep_param_name="",           # <-- no dummy injected
    distance_kwargs_base={},
    gif_path="recon_ambient.gif",
    fps=1,
)
###################################################################################################

def run_target_m_sweep_fermat(
    m_list: List[float],
    templates: List[Shape],
    n_target: int,
    p_fermat: float,
    target_a: float = 1.0,
    target_b: float = 1.0,
    graph: str = "cycle",
    k: int = 10,
    gif_path: str = "recon_fermat_target_m_sweep.gif",
    fps: int = 1,
    mds_random_state: int = 42,
):
    """
    Build a GIF where the TARGET is a superellipse with varying exponent m,
    and internal distances are p-Fermat (fixed p_fermat).
    Templates remain fixed.

    Frames: m in m_list
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=mds_random_state)

    # ---- Precompute template distance matrices ONCE (since p is fixed) ----
    matrix_temp_list = []
    measure_temp_list = []
    for s in templates:
        Dtemp = build_distance(
            s.points,
            mode="fermat",
            p=p_fermat,
            graph=graph,
            k=k,
            assume_ordered_cycle=s.ordered_cycle,
        )
        matrix_temp_list.append(Dtemp)
        measure_temp_list.append(s.measure)

    embeddings = []
    lambdas_all = []
    gw_all = []
    targets = []  # store target shapes for plotting / sizes

    for m in m_list:
        print(f"Frame: m={m} (Fermat p={p_fermat})")

        # Build target shape for this frame
        P = build_superellipse(n=n_target, m=m, a=target_a, b=target_b)
        b = make_uniform_measure(n_target)
        target_shape = Shape(name=f"superellipse_m={m}", points=P, measure=b, ordered_cycle=True)
        targets.append(target_shape)

        # Target Fermat distance matrix
        B = build_distance(
            P,
            mode="fermat",
            p=p_fermat,
            graph=graph,
            k=k,
            assume_ordered_cycle=True,
        )

        # Reconstruction + lambdas
        B_recon, lambdas_est = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, B, b
        )
        B_recon = 0.5 * (B_recon + B_recon.T)
        np.fill_diagonal(B_recon, 0.0)

        # GW distance between B and B_recon (same measure b)
        gromov_distance = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
        gw_dist = gromov_distance["gw_dist"]

        # MDS embedding of reconstruction
        X = mds.fit_transform(B_recon)

        embeddings.append(X)
        lambdas_all.append(lambdas_est)
        gw_all.append(gw_dist)

    # ---- Align embeddings across frames ----
    ref_idx = len(m_list) // 2
    Xref = embeddings[ref_idx]
    embeddings = [align_to_reference(X, Xref) for X in embeddings]

    # ---- Consistent axis limits ----
    all_pts = np.vstack(embeddings)
    range_x = np.ptp(all_pts[:, 0])
    range_y = np.ptp(all_pts[:, 1])
    scale = max(range_x, range_y, 1e-12)
    pad = 0.05 * scale

    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

    # ---- Animation (initialize with first frame) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(embeddings[0][:, 0], embeddings[0][:, 1], s=targets[0].measure * 350)
    title = ax.set_title("")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    def init():
        sc.set_offsets(embeddings[0])
        sc.set_sizes(targets[0].measure * 350)
        title.set_text("")
        return sc, title

    def update(i):
        sc.set_offsets(embeddings[i])
        sc.set_sizes(targets[i].measure * 350)

        m_val = m_list[i]
        lam = lambdas_all[i]
        gw = gw_all[i]
        title.set_text(
            f"MDS(recon) | dist=fermat(p={p_fermat}) | target m={m_val} | GW={gw:.4g} | lambdas={np.array(lam)}"
        )
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(m_list), init_func=init, blit=True)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to: {gif_path}")


m_list = [2, 2.5, 3, 4, 6, 10]
p_list = [1, 1.5, 2, 4, 10]

for p in p_list:
    run_target_m_sweep_fermat(
        m_list=m_list,
        templates=templates,
        n_target=120,
        p_fermat=p,
        graph="cycle",
        gif_path=f"recon_fermat_target_m_sweep_p{p}.gif",
        fps=1,
    )



###################################################################################################
## PLOT TEMPLATES AND TARGET  #####################################################################
###################################################################################################

def plot_templates_and_target(templates: List[Shape], target: Shape,
                              save_path: str = "templates_and_target.png",
                              show: bool = True) -> None:
    """
    Plot each template + the target at the end of the script.
    Saves a PNG by default.
    """
    shapes = templates + [target]
    n = len(shapes)

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, shapes):
        P = s.points

        # close the curve for nicer visualization if it's an ordered cycle
        if s.ordered_cycle and P.shape[0] > 1:
            Pplot = np.vstack([P, P[0]])
        else:
            Pplot = P

        ax.plot(Pplot[:, 0], Pplot[:, 1], "-o", markersize=2, linewidth=1)
        ax.set_title(s.name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved shapes plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# --- Plot templates + target (static) ---
plot_templates_and_target(templates, target_super, save_path="templates_and_target.png", show=True)
