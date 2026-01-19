# Exploration on using internal / geodesic distances
#
# We consider: geodesic templates; target Fermat(p)

import numpy as np
import pandas as pd
import ot
import utils  # must provide: utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)

from scipy.spatial.distance import cdist


# =========================
# Parameters
# =========================

P_LIST = [1, 1.25, 1.5, 1.75, 2, 4, 6, 8, 10]
M_LIST = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 4, 6, 8]

N_CIRCLE = 120
N_SQUARE = 120
N_TARGET = 120

NORMALIZE_MODE = "diameter"   # options: "diameter", "mean", None
CORNER_DENSITY = 1.0          # square corner clustering strength

OUT_XLSX = "lambda_gw_table.xlsx"


# =========================
# Helpers
# =========================

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


# =========================
# Shape builders
# =========================

def build_circle_points(n: int = 120, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    P = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1)
    return P, theta

def _corner_dense_param(u: np.ndarray, strength: float = 1.0) -> np.ndarray:
    t = 0.5 * (1.0 - np.cos(np.pi * u))  # clusters near endpoints
    if strength != 1.0:
        t = t ** strength
        t = np.clip(t, 0.0, 1.0 - 1e-12)
    return t

def build_square_corner_dense(n_points: int = 120, half_width: float = 1.0, strength: float = 1.0) -> np.ndarray:
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
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * (np.abs(ct) ** (2.0 / m))
    y = b * np.sign(st) * (np.abs(st) ** (2.0 / m))
    return np.stack([x, y], axis=1)


# =========================
# Distances for Table 1
# =========================

def circle_geodesic_exact_from_angles(theta: np.ndarray, radius: float = 1.0) -> np.ndarray:
    dtheta = np.abs(theta[:, None] - theta[None, :])
    dtheta = np.minimum(dtheta, 2*np.pi - dtheta)
    D = radius * dtheta
    return symmetrize_zero_diag(D)

def cycle_geodesic_distance(P: np.ndarray) -> np.ndarray:
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


# =========================
# Table 1 computation
# =========================

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


# =========================
# Build templates
# =========================

P_circle, theta_circle = build_circle_points(n=N_CIRCLE, radius=1.0)
mu_circle = uniform_measure(N_CIRCLE)

P_square_corner = build_square_corner_dense(
    n_points=N_SQUARE, half_width=1.0, strength=CORNER_DENSITY
)
mu_square_corner = uniform_measure(P_square_corner.shape[0])

A1_geo = circle_geodesic_exact_from_angles(theta_circle, radius=1.0)
A2_geo = cycle_geodesic_distance(P_square_corner)


# =========================
# Run Table and export
# =========================

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

with pd.ExcelWriter(OUT_XLSX) as writer:
    df1_text.to_excel(writer, sheet_name="Table1_geodesic_templates")

print(f"Saved: {OUT_XLSX}")
