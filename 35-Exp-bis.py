# Exploration on using internal / geodesic distances

import numpy as np
import pandas as pd

import ot
import utils  # must provide: utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


# %% =========================
# Parameters
# ============================

P_LIST = [1, 1.25, 1.5, 1.75, 2, 4, 6, 8, 10]
M_LIST = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 4, 6, 8]

N_CIRCLE = 120
N_SQUARE = 120
N_TARGET = 120

# Normalization makes lambdas comparable across shapes/scales and across (p,m).
# Set to None to disable.
NORMALIZE_MODE = "diameter"   # options: "diameter", "mean", None

# corner density strength for square in Table 1 (bigger => more corner clustering)
CORNER_DENSITY = 1.0  # 1.0 is already good with cosine map; increase for stronger corner mass

# %% =========================
# Experiment flags (toggle)
# ============================

RUN_EXP1_FAIR_INTRINSIC_P = True     # (1) templates also use Fermat(p)
RUN_EXP2_GEODESIC_TARGET = True      # (2) target geodesic; compare template geodesic vs ambient
RUN_EXP3_SAMPLING_ROBUSTNESS = True  # (3) vary N and corner density for a fixed (p,m)
RUN_EXP4_NON_EUCLIDEAN_MANIFOLDS = True  # (4) kNN geodesics on sphere/ellipsoid demo


# %% =========================
# Helpers: measures and normalization
# ============================

def uniform_measure(n: int) -> np.ndarray:
    return np.ones(n) / float(n)


def normalize_distance_matrix(D: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return D
    if mode == "diameter":
        denom = float(np.max(D))
    elif mode == "mean":
        denom = float(np.mean(D))
    else:
        raise ValueError("normalize mode must be 'diameter', 'mean', or None")

    if denom <= 0 or not np.isfinite(denom):
        return D
    return D / denom


def symmetrize_zero_diag(D: np.ndarray) -> np.ndarray:
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# %% =========================
# Shape builders
# ============================

def build_circle_points(n: int = 120, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      points (n,2)
      angles (n,)
    """
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    P = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1)
    return P, theta


def build_square_uniform(n_points: int = 120, half_width: float = 1.0) -> np.ndarray:
    """
    Uniform sampling along boundary of square [-a,a]^2 in cyclic order.
    """
    n_side = max(1, n_points // 4)
    n_points = 4 * n_side
    a = half_width

    u = np.linspace(0, 1, n_side, endpoint=False)
    x = -a + 2*a*u

    bottom = np.stack([x, -a*np.ones_like(x)], axis=1)
    right  = np.stack([a*np.ones_like(x), x], axis=1)
    top    = np.stack([x[::-1], a*np.ones_like(x)], axis=1)
    left   = np.stack([-a*np.ones_like(x), x[::-1]], axis=1)

    return np.concatenate([bottom, right, top, left], axis=0)


def _corner_dense_param(u: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Map uniform u in [0,1) to t in [0,1) with higher density near endpoints.
    """
    t = 0.5 * (1.0 - np.cos(np.pi * u))  # clusters near 0,1
    if strength != 1.0:
        t = t ** strength
        t = np.clip(t, 0.0, 1.0 - 1e-12)
    return t


def build_square_corner_dense(n_points: int = 120, half_width: float = 1.0, strength: float = 1.0) -> np.ndarray:
    """
    Corner-dense sampling along square boundary [-a,a]^2 in cyclic order.
    Uniform measure on these points => more mass near corners.
    """
    n_side = max(1, n_points // 4)
    n_points = 4 * n_side
    a = half_width

    u = np.linspace(0, 1, n_side, endpoint=False)
    t = _corner_dense_param(u, strength=strength)
    x = -a + 2*a*t

    bottom = np.stack([x, -a*np.ones_like(x)], axis=1)
    right  = np.stack([a*np.ones_like(x), x], axis=1)
    top    = np.stack([x[::-1], a*np.ones_like(x)], axis=1)
    left   = np.stack([-a*np.ones_like(x), x[::-1]], axis=1)

    return np.concatenate([bottom, right, top, left], axis=0)


def build_superellipse(n: int = 120, m: float = 2.0, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """
    Superellipse |x|^m + |y|^m = 1 via param:
      x = a * sgn(cos t) * |cos t|^{2/m}
      y = b * sgn(sin t) * |sin t|^{2/m}
    Works for any m>0.
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * (np.abs(ct) ** (2.0 / m))
    y = b * np.sign(st) * (np.abs(st) ** (2.0 / m))
    return np.stack([x, y], axis=1)


# %% =========================
# Distances: circle geodesic, cycle geodesic, cycle Fermat, ambient
# ============================

def circle_geodesic_exact_from_angles(theta: np.ndarray, radius: float = 1.0) -> np.ndarray:
    dtheta = np.abs(theta[:, None] - theta[None, :])
    dtheta = np.minimum(dtheta, 2*np.pi - dtheta)
    D = radius * dtheta
    return symmetrize_zero_diag(D)


def cycle_geodesic_distance(P: np.ndarray) -> np.ndarray:
    """
    Geodesic on ordered closed contour using neighbor edge lengths.
    """
    n = P.shape[0]
    L = np.linalg.norm(P[(np.arange(n)+1) % n] - P[np.arange(n)], axis=1)
    S = np.concatenate([[0.0], np.cumsum(L)])
    total = S[-1]

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            forward = S[j] - S[i]
            backward = total - forward
            dij = min(forward, backward)
            D[i, j] = dij
            D[j, i] = dij
    return symmetrize_zero_diag(D)


def cycle_fermat_distance(P: np.ndarray, p: float) -> np.ndarray:
    """
    Fermat(p) on ordered cycle graph.
    """
    n = P.shape[0]
    L = np.linalg.norm(P[(np.arange(n)+1) % n] - P[np.arange(n)], axis=1)
    W = L ** p
    S = np.concatenate([[0.0], np.cumsum(W)])
    total = S[-1]

    D = np.zeros((n, n), dtype=float)
    invp = 1.0 / p
    for i in range(n):
        for j in range(i+1, n):
            forward = S[j] - S[i]
            backward = total - forward
            dij = min(forward, backward) ** invp
            D[i, j] = dij
            D[j, i] = dij
    return symmetrize_zero_diag(D)


def ambient_distance(P: np.ndarray) -> np.ndarray:
    return symmetrize_zero_diag(cdist(P, P, metric="euclidean"))


# %% =========================
# Core: compute (p,m)-grid table with FIXED template distance matrices
# Target uses Fermat(p)
# ============================

def compute_lambda_gw_table(
    A1: np.ndarray,
    A2: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    p_list: list[float],
    m_list: list[float],
    n_target: int,
    normalize_mode: str | None = "diameter",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    A1n = normalize_distance_matrix(A1, normalize_mode)
    A2n = normalize_distance_matrix(A2, normalize_mode)

    matrix_temp_list = [A1n, A2n]
    measure_temp_list = [mu1, mu2]

    df_text = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_lambda = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_gw = pd.DataFrame(index=p_list, columns=m_list, dtype=float)

    b = uniform_measure(n_target)

    for p in p_list:
        for m in m_list:
            P_target = build_superellipse(n=n_target, m=m, a=1.0, b=1.0)

            B = cycle_fermat_distance(P_target, p=p)
            B = normalize_distance_matrix(B, normalize_mode)

            B_recon, lambdas_est = utils.get_lambdas_constraints(
                matrix_temp_list, measure_temp_list, B, b
            )
            B_recon = symmetrize_zero_diag(B_recon)
            B_recon = normalize_distance_matrix(B_recon, normalize_mode)

            try:
                gw = ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss")
                gw = float(gw)
            except Exception:
                log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
                gw = float(log["gw_dist"])

            lam = np.array(lambdas_est, dtype=float).ravel()
            lam1, lam2 = float(lam[0]), float(lam[1])

            df_lambda.loc[p, m] = (lam1, lam2)
            df_gw.loc[p, m] = gw
            df_text.loc[p, m] = f"λ=({lam1:.4g},{lam2:.4g}), GW={gw:.4g}"

    return df_text, df_lambda, df_gw


# %% =========================
# Table 3 helper: ambient target (no p) over m
# ============================

def compute_lambda_gw_table_ambient_target(
    A1: np.ndarray,
    A2: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    m_list: list[float],
    n_target: int,
    normalize_mode: str | None = "diameter",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    A1n = normalize_distance_matrix(A1, normalize_mode)
    A2n = normalize_distance_matrix(A2, normalize_mode)

    matrix_temp_list = [A1n, A2n]
    measure_temp_list = [mu1, mu2]

    row_label = "ambient"
    df_text = pd.DataFrame(index=[row_label], columns=m_list, dtype=object)
    df_lambda = pd.DataFrame(index=[row_label], columns=m_list, dtype=object)
    df_gw = pd.DataFrame(index=[row_label], columns=m_list, dtype=float)

    b = uniform_measure(n_target)

    for m in m_list:
        P_target = build_superellipse(n=n_target, m=m, a=1.0, b=1.0)

        B = ambient_distance(P_target)
        B = normalize_distance_matrix(B, normalize_mode)

        B_recon, lambdas_est = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, B, b
        )
        B_recon = symmetrize_zero_diag(B_recon)
        B_recon = normalize_distance_matrix(B_recon, normalize_mode)

        try:
            gw = ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss")
            gw = float(gw)
        except Exception:
            log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
            gw = float(log["gw_dist"])

        lam = np.array(lambdas_est, dtype=float).ravel()
        lam1, lam2 = float(lam[0]), float(lam[1])

        df_lambda.loc[row_label, m] = (lam1, lam2)
        df_gw.loc[row_label, m] = gw
        df_text.loc[row_label, m] = f"λ=({lam1:.4g},{lam2:.4g}), GW={gw:.4g}"

    return df_text, df_lambda, df_gw


# %% =========================
# Build template point clouds + template distances for Tables 1-3
# ============================

# Circle template (uniform)
P_circle, theta_circle = build_circle_points(n=N_CIRCLE, radius=1.0)
mu_circle = uniform_measure(N_CIRCLE)

# Square templates
P_square_uniform = build_square_uniform(n_points=N_SQUARE, half_width=1.0)
mu_square_uniform = uniform_measure(P_square_uniform.shape[0])

P_square_corner = build_square_corner_dense(n_points=N_SQUARE, half_width=1.0, strength=CORNER_DENSITY)
mu_square_corner = uniform_measure(P_square_corner.shape[0])

# Table 1 templates: geodesic circle (exact), geodesic square (corner-dense)
A1_geo = circle_geodesic_exact_from_angles(theta_circle, radius=1.0)
A2_geo = cycle_geodesic_distance(P_square_corner)

# Table 2 templates: ambient circle + ambient square (uniform)
A1_amb = ambient_distance(P_circle)
A2_amb = ambient_distance(P_square_uniform)


# %% =========================
# Table 1: geodesic templates; target Fermat(p)
# ============================

df1_text, df1_lambda, df1_gw = compute_lambda_gw_table(
    A1=A1_geo,
    A2=A2_geo,
    mu1=mu_circle,
    mu2=mu_square_corner,
    p_list=P_LIST,
    m_list=M_LIST,
    n_target=N_TARGET,
    normalize_mode=NORMALIZE_MODE,
)

df1_text


# %% =========================
# Table 2: ambient templates; target Fermat(p)
# ============================

df2_text, df2_lambda, df2_gw = compute_lambda_gw_table(
    A1=A1_amb,
    A2=A2_amb,
    mu1=mu_circle,
    mu2=mu_square_uniform,
    p_list=P_LIST,
    m_list=M_LIST,
    n_target=N_TARGET,
    normalize_mode=NORMALIZE_MODE,
)

df2_text


# %% =========================
# Table 3: ambient templates AND ambient target; only over m
# ============================

df3_text, df3_lambda, df3_gw = compute_lambda_gw_table_ambient_target(
    A1=A1_amb,
    A2=A2_amb,
    mu1=mu_circle,
    mu2=mu_square_uniform,
    m_list=M_LIST,
    n_target=N_TARGET,
    normalize_mode=NORMALIZE_MODE,
)

df3_text


# %% ======================================================================
# TABLE 4: Templates are circles with different sampling/metric choices
#   Template 1: uniform circle Fermat(p) (depends on row p)
#   Template 2: 4-vonMises sampled circle Fermat(1) (fixed)
# Target: superellipse Fermat(p)
# ======================================================================

def build_circle_vonmises_mixture(
    n: int,
    mus: list[float],
    kappas: list[float] | float,
    weights: list[float] | None = None,
    seed: int = 0,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    K = len(mus)

    if isinstance(kappas, (int, float)):
        kappas = [float(kappas)] * K
    else:
        kappas = [float(x) for x in kappas]
        if len(kappas) != K:
            raise ValueError("kappas must be a scalar or same length as mus.")

    if weights is None:
        weights = [1.0 / K] * K
    else:
        weights = np.array(weights, dtype=float)
        weights = (weights / weights.sum()).tolist()

    comps = rng.choice(K, size=n, p=weights)
    theta = np.empty(n, dtype=float)

    for k in range(K):
        idx = np.where(comps == k)[0]
        if idx.size > 0:
            theta[idx] = rng.vonmises(mu=mus[k], kappa=kappas[k], size=idx.size)

    theta = np.mod(theta, 2*np.pi)
    order = np.argsort(theta)
    theta = theta[order]

    P = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1)
    return P, theta


def compute_table4_circle_circle(
    p_list: list[float],
    m_list: list[float],
    n_uniform_circle: int,
    n_vonmises_circle: int,
    n_target: int,
    normalize_mode: str | None = "diameter",
    seed: int = 0,
    kappa: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Template 1: uniform circle
    P_circ_u, _ = build_circle_points(n=n_uniform_circle, radius=1.0)
    mu_u = uniform_measure(n_uniform_circle)

    # Template 2: vonMises-mixture circle
    mus = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
    P_circ_vm, _ = build_circle_vonmises_mixture(
        n=n_vonmises_circle,
        mus=mus,
        kappas=kappa,
        weights=None,
        seed=seed,
        radius=1.0,
    )
    mu_vm = uniform_measure(n_vonmises_circle)

    # A2 fixed: Fermat(1)
    A2 = cycle_fermat_distance(P_circ_vm, p=1.0)
    A2 = normalize_distance_matrix(A2, normalize_mode)

    df_text = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_lambda = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_gw = pd.DataFrame(index=p_list, columns=m_list, dtype=float)

    b = uniform_measure(n_target)

    for p in p_list:
        A1 = cycle_fermat_distance(P_circ_u, p=p)
        A1 = normalize_distance_matrix(A1, normalize_mode)

        matrix_temp_list = [A1, A2]
        measure_temp_list = [mu_u, mu_vm]

        for m in m_list:
            P_target = build_superellipse(n=n_target, m=m, a=1.0, b=1.0)
            B = cycle_fermat_distance(P_target, p=p)
            B = normalize_distance_matrix(B, normalize_mode)

            B_recon, lambdas_est = utils.get_lambdas_constraints(
                matrix_temp_list, measure_temp_list, B, b
            )
            B_recon = symmetrize_zero_diag(B_recon)
            B_recon = normalize_distance_matrix(B_recon, normalize_mode)

            try:
                gw = float(ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss"))
            except Exception:
                log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
                gw = float(log["gw_dist"])

            lam = np.array(lambdas_est, dtype=float).ravel()
            lam1, lam2 = float(lam[0]), float(lam[1])

            df_lambda.loc[p, m] = (lam1, lam2)
            df_gw.loc[p, m] = gw
            df_text.loc[p, m] = f"λ=({lam1:.4g},{lam2:.4g}), GW={gw:.4g}"

    return df_text, df_lambda, df_gw


df4_text, df4_lambda, df4_gw = compute_table4_circle_circle(
    p_list=P_LIST,
    m_list=M_LIST,
    n_uniform_circle=N_CIRCLE,
    n_vonmises_circle=N_CIRCLE,
    n_target=N_TARGET,
    normalize_mode=NORMALIZE_MODE,
    seed=0,
    kappa=20.0,
)

df4_text


# %% ======================================================================
# EXPERIMENT (1): "Fair intrinsic" across p
# Templates use Fermat(p) as well; Target uses Fermat(p)
# ======================================================================

def compute_lambda_gw_table_templates_depend_on_p(
    p_list: list[float],
    m_list: list[float],
    n_circle: int,
    n_square: int,
    n_target: int,
    use_corner_dense_square: bool = False,
    corner_density: float = 1.0,
    normalize_mode: str | None = "diameter",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    P_circle_u, _ = build_circle_points(n=n_circle, radius=1.0)
    mu_circle_u = uniform_measure(n_circle)

    if use_corner_dense_square:
        P_square = build_square_corner_dense(n_points=n_square, half_width=1.0, strength=corner_density)
    else:
        P_square = build_square_uniform(n_points=n_square, half_width=1.0)
    mu_square = uniform_measure(P_square.shape[0])

    df_text = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_lambda = pd.DataFrame(index=p_list, columns=m_list, dtype=object)
    df_gw = pd.DataFrame(index=p_list, columns=m_list, dtype=float)

    b = uniform_measure(n_target)

    for p in p_list:
        A1 = normalize_distance_matrix(cycle_fermat_distance(P_circle_u, p=p), normalize_mode)
        A2 = normalize_distance_matrix(cycle_fermat_distance(P_square, p=p), normalize_mode)

        matrix_temp_list = [A1, A2]
        measure_temp_list = [mu_circle_u, mu_square]

        for m in m_list:
            P_target = build_superellipse(n=n_target, m=m, a=1.0, b=1.0)
            B = normalize_distance_matrix(cycle_fermat_distance(P_target, p=p), normalize_mode)

            B_recon, lambdas_est = utils.get_lambdas_constraints(
                matrix_temp_list, measure_temp_list, B, b
            )
            B_recon = normalize_distance_matrix(symmetrize_zero_diag(B_recon), normalize_mode)

            try:
                gw = float(ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss"))
            except Exception:
                log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
                gw = float(log["gw_dist"])

            lam = np.array(lambdas_est, dtype=float).ravel()
            lam1, lam2 = float(lam[0]), float(lam[1])

            df_lambda.loc[p, m] = (lam1, lam2)
            df_gw.loc[p, m] = gw
            df_text.loc[p, m] = f"λ=({lam1:.4g},{lam2:.4g}), GW={gw:.4g}"

    return df_text, df_lambda, df_gw


if RUN_EXP1_FAIR_INTRINSIC_P:
    df5_text, df5_lambda, df5_gw = compute_lambda_gw_table_templates_depend_on_p(
        p_list=P_LIST,
        m_list=M_LIST,
        n_circle=N_CIRCLE,
        n_square=N_SQUARE,
        n_target=N_TARGET,
        use_corner_dense_square=False,  # True if you want discrete mass near corners
        corner_density=CORNER_DENSITY,
        normalize_mode=NORMALIZE_MODE,
    )
    df5_text


# %% ======================================================================
# EXPERIMENT (2): Target GEODESIC; compare templates geodesic vs templates ambient
# Produces a 2-row table over m.
# ======================================================================

def compute_one_row_table_geodesic_target(
    A1: np.ndarray,
    A2: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    m_list: list[float],
    n_target: int,
    normalize_mode: str | None = "diameter",
    row_label: str = "row",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    A1n = normalize_distance_matrix(A1, normalize_mode)
    A2n = normalize_distance_matrix(A2, normalize_mode)

    matrix_temp_list = [A1n, A2n]
    measure_temp_list = [mu1, mu2]

    df_text = pd.DataFrame(index=[row_label], columns=m_list, dtype=object)
    df_lambda = pd.DataFrame(index=[row_label], columns=m_list, dtype=object)
    df_gw = pd.DataFrame(index=[row_label], columns=m_list, dtype=float)

    b = uniform_measure(n_target)

    for m in m_list:
        P_target = build_superellipse(n=n_target, m=m, a=1.0, b=1.0)

        B = cycle_geodesic_distance(P_target)
        B = normalize_distance_matrix(B, normalize_mode)

        B_recon, lambdas_est = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, B, b
        )
        B_recon = normalize_distance_matrix(symmetrize_zero_diag(B_recon), normalize_mode)

        try:
            gw = float(ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss"))
        except Exception:
            log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
            gw = float(log["gw_dist"])

        lam = np.array(lambdas_est, dtype=float).ravel()
        lam1, lam2 = float(lam[0]), float(lam[1])

        df_lambda.loc[row_label, m] = (lam1, lam2)
        df_gw.loc[row_label, m] = gw
        df_text.loc[row_label, m] = f"λ=({lam1:.4g},{lam2:.4g}), GW={gw:.4g}"

    return df_text, df_lambda, df_gw


if RUN_EXP2_GEODESIC_TARGET:
    P_c, theta_c = build_circle_points(n=N_CIRCLE, radius=1.0)
    P_s = build_square_uniform(n_points=N_SQUARE, half_width=1.0)

    mu_c = uniform_measure(N_CIRCLE)
    mu_s = uniform_measure(P_s.shape[0])

    A1_geo_u = circle_geodesic_exact_from_angles(theta_c, radius=1.0)
    A2_geo_u = cycle_geodesic_distance(P_s)

    df6a_text, df6a_lambda, df6a_gw = compute_one_row_table_geodesic_target(
        A1=A1_geo_u, A2=A2_geo_u,
        mu1=mu_c, mu2=mu_s,
        m_list=M_LIST, n_target=N_TARGET,
        normalize_mode=NORMALIZE_MODE,
        row_label="templates_geodesic",
    )

    A1_amb_u = ambient_distance(P_c)
    A2_amb_u = ambient_distance(P_s)

    df6b_text, df6b_lambda, df6b_gw = compute_one_row_table_geodesic_target(
        A1=A1_amb_u, A2=A2_amb_u,
        mu1=mu_c, mu2=mu_s,
        m_list=M_LIST, n_target=N_TARGET,
        normalize_mode=NORMALIZE_MODE,
        row_label="templates_ambient",
    )

    df6_text = pd.concat([df6a_text, df6b_text], axis=0)
    df6_text


# %% ======================================================================
# EXPERIMENT (3): Sampling robustness
# Vary N and corner density for a fixed (p,m); output a small summary DataFrame.
# ======================================================================

def sampling_robustness_study(
    N_list: list[int],
    corner_density_list: list[float],
    p: float,
    m: float,
    normalize_mode: str | None = "diameter",
) -> pd.DataFrame:

    rows = []
    for N in N_list:
        P_c, theta_c = build_circle_points(n=N, radius=1.0)
        mu_c = uniform_measure(N)
        A_c = normalize_distance_matrix(circle_geodesic_exact_from_angles(theta_c, radius=1.0), normalize_mode)

        for cd in corner_density_list:
            P_s = build_square_corner_dense(n_points=N, half_width=1.0, strength=cd)
            mu_s = uniform_measure(P_s.shape[0])
            A_s = normalize_distance_matrix(cycle_geodesic_distance(P_s), normalize_mode)

            P_t = build_superellipse(n=N, m=m, a=1.0, b=1.0)
            b = uniform_measure(N)
            B = normalize_distance_matrix(cycle_fermat_distance(P_t, p=p), normalize_mode)

            B_recon, lambdas_est = utils.get_lambdas_constraints([A_c, A_s], [mu_c, mu_s], B, b)
            B_recon = normalize_distance_matrix(symmetrize_zero_diag(B_recon), normalize_mode)

            try:
                gw = float(ot.gromov.gromov_wasserstein2(B, B_recon, b, b, loss_fun="square_loss"))
            except Exception:
                log = ot.gromov.gromov_wasserstein(B, B_recon, b, b, log=True)[1]
                gw = float(log["gw_dist"])

            lam = np.array(lambdas_est, dtype=float).ravel()
            lam1, lam2 = float(lam[0]), float(lam[1])

            rows.append({
                "N": N,
                "corner_density": cd,
                "p_target": p,
                "m_target": m,
                "lambda1": lam1,
                "lambda2": lam2,
                "GW": gw,
            })

    return pd.DataFrame(rows)


if RUN_EXP3_SAMPLING_ROBUSTNESS:
    df7 = sampling_robustness_study(
        N_list=[60, 120, 240],
        corner_density_list=[1.0, 2.0, 4.0],
        p=1.0,
        m=6.0,
        normalize_mode=NORMALIZE_MODE,
    )
    df7


# %% ======================================================================
# EXPERIMENT (4): Non-Euclidean manifolds via kNN geodesics (sphere vs ellipsoid)
# ======================================================================

def sample_sphere(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def sample_ellipsoid(n: int, axes=(1.0, 1.0, 0.6), seed: int = 0) -> np.ndarray:
    X = sample_sphere(n, seed=seed)
    return X * np.array(axes, dtype=float)[None, :]


def knn_geodesic_distance(P: np.ndarray, k: int = 10) -> np.ndarray:
    D = cdist(P, P, metric="euclidean")
    n = D.shape[0]
    rows, cols, data = [], [], []

    for i in range(n):
        nn = np.argsort(D[i])[1:k+1]
        for j in nn:
            rows.append(i); cols.append(j); data.append(D[i, j])

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)

    sp = shortest_path(W, directed=False, method="D")

    if not np.isfinite(sp).all():
        finite = sp[np.isfinite(sp)]
        big = 2.0 * finite.max() if finite.size else 1.0
        sp = np.where(np.isfinite(sp), sp, big)

    return symmetrize_zero_diag(sp)


def manifold_demo_table(
    n: int = 200,
    k: int = 12,
    t_list: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    normalize_mode: str | None = "diameter",
) -> pd.DataFrame:

    mu = uniform_measure(n)

    # Templates
    P1 = sample_sphere(n, seed=1)
    P2 = sample_ellipsoid(n, axes=(1.0, 1.0, 0.6), seed=2)

    A1 = normalize_distance_matrix(knn_geodesic_distance(P1, k=k), normalize_mode)
    A2 = normalize_distance_matrix(knn_geodesic_distance(P2, k=k), normalize_mode)

    rows = []
    for t in t_list:
        c = 1.0 - 0.4*t
        Pt = sample_ellipsoid(n, axes=(1.0, 1.0, c), seed=3)

        B = normalize_distance_matrix(knn_geodesic_distance(Pt, k=k), normalize_mode)

        B_recon, lambdas_est = utils.get_lambdas_constraints([A1, A2], [mu, mu], B, mu)
        B_recon = normalize_distance_matrix(symmetrize_zero_diag(B_recon), normalize_mode)

        try:
            gw = float(ot.gromov.gromov_wasserstein2(B, B_recon, mu, mu, loss_fun="square_loss"))
        except Exception:
            log = ot.gromov.gromov_wasserstein(B, B_recon, mu, mu, log=True)[1]
            gw = float(log["gw_dist"])

        lam = np.array(lambdas_est, dtype=float).ravel()
        lam1, lam2 = float(lam[0]), float(lam[1])

        rows.append({"t": t, "axes_c": c, "lambda1": lam1, "lambda2": lam2, "GW": gw})

    return pd.DataFrame(rows)


if RUN_EXP4_NON_EUCLIDEAN_MANIFOLDS:
    df8 = manifold_demo_table(
        n=200,
        k=12,
        t_list=[0.0, 0.25, 0.5, 0.75, 1.0],
        normalize_mode=NORMALIZE_MODE,
    )
    df8


# %% =========================
# Export to Excel
# ============================

OUT_XLSX = "lambda_gw_tables_different_internal_distances_FULL.xlsx"

with pd.ExcelWriter(OUT_XLSX) as writer:
    df1_text.to_excel(writer, sheet_name="Table1_geodesic_templates")
    df2_text.to_excel(writer, sheet_name="Table2_ambient_templates")
    df3_text.to_excel(writer, sheet_name="Table3_ambient_only_m")
    df4_text.to_excel(writer, sheet_name="Table4_circle_circle_vm")

    if RUN_EXP1_FAIR_INTRINSIC_P:
        df5_text.to_excel(writer, sheet_name="Table5_templates_Fermat_p")

    if RUN_EXP2_GEODESIC_TARGET:
        df6_text.to_excel(writer, sheet_name="Table6_target_geodesic_compare")

    if RUN_EXP3_SAMPLING_ROBUSTNESS:
        df7.to_excel(writer, sheet_name="Exp3_sampling_robustness", index=False)

    if RUN_EXP4_NON_EUCLIDEAN_MANIFOLDS:
        df8.to_excel(writer, sheet_name="Exp4_manifold_demo", index=False)

print(f"Saved: {OUT_XLSX}")

