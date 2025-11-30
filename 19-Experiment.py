import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.spatial
import ot
import torch

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------
# Helpers for shapes
# ---------------------------------------------------
def load_binary_shape(path, thresh=0.95):
    """Load a grayscale/binary PNG and return the dark-pixel coordinates."""
    img = plt.imread(path).astype(np.float64)
    img = img[..., 2]    # take one channel (your original behavior)

    pts = []
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            # keep non-white pixels (threshold 0.95 like your code)
            if img[i, j] < thresh:
                pts.append([j, H - i])
    return np.array(pts)


def compute_cost_matrix(X):
    """Pairwise Euclidean distances, normalized."""
    C = scipy.spatial.distance.cdist(X, X)
    return C / C.max()


# ----------------------------
# Helper: grad of GW^2 wrt Cb given gamma
# ----------------------------
def gw_grad_Cb(Cb, Ck, gamma):
    """
    Cb: (M, M) torch tensor
    Ck: (n_k, n_k) torch tensor
    gamma: (M, n_k) torch tensor
    returns grad_Cb: (M, M)
    """
    # Cb[i,i'] - Ck[j,j']
    # shape Cb: (M,M), Ck: (n_k,n_k), gamma: (M,n_k)
    # grad_Cb[i,i'] = 2 * sum_{j,j'} (Cb[i,i'] - Ck[j,j']) * gamma[i,j]*gamma[i',j']

    # Expand shapes for broadcasting:
    # Cb -> (M,M,1,1)
    # Ck -> (1,1,nk,nk)
    diff = Cb.unsqueeze(2).unsqueeze(3) - Ck.unsqueeze(0).unsqueeze(1)  # (M,M,nk,nk)

    # gamma_i_j * gamma_i'_j'
    G1 = gamma.unsqueeze(1).unsqueeze(3)      # (M,1,nk,1)
    G2 = gamma.unsqueeze(0).unsqueeze(2)      # (1,M,1,nk)
    gg = G1 * G2                              # (M,M,nk,nk)

    grad = 2.0 * (diff * gg).sum(dim=(2, 3))  # sum over j,j'
    return grad  # (M,M)


# ----------------------------
# Custom autograd Function for OUTER GW loss
# ----------------------------
class GWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Cb, C_target):
        Cb_np = Cb.detach().cpu().numpy()
        Ct_np = C_target.detach().cpu().numpy()

        p_b = ot.unif(Cb_np.shape[0])
        p_t = ot.unif(Ct_np.shape[0])

        # Compute coupling with correct loss_fun
        pi_star = ot.gromov.gromov_wasserstein(
            Cb_np, Ct_np, p_b, p_t, loss_fun="square_loss"
        )

        # Compute loss (NO "square_loss" here)
        loss = ot.gromov.gwloss(Cb_np, Ct_np, pi_star)

        ctx.save_for_backward(
            Cb,
            C_target,
            torch.from_numpy(pi_star).float().to(Cb.device)
        )

        return Cb.new_tensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            Cb, C_target, pi_star = ctx.saved_tensors
            grad_Cb = gw_grad_Cb(Cb, C_target, pi_star)
            grad_Cb = grad_output * grad_Cb
        return grad_Cb, None


# ----------------------------
# Helper: compute GW coupling with POT without gradient (for barycenter)
# ----------------------------
def gw_coupling_pot(Cb_torch, Ck_torch, p_b_torch, pk_torch):
    """
    Cb_torch: (M, M) torch tensor
    Ck_torch: (n_k, n_k) torch tensor
    p_b_torch: (M,) torch tensor
    pk_torch: (n_k,) torch tensor
    """
    Cb_np = Cb_torch.detach().cpu().numpy()
    Ck_np = Ck_torch.detach().cpu().numpy()
    p_b_np = p_b_torch.detach().cpu().numpy()
    pk_np = pk_torch.detach().cpu().numpy()

    gamma = ot.gromov.gromov_wasserstein(
        Cb_np, Ck_np, p_b_np, pk_np, loss_fun="square_loss"
    )
    gamma_torch = torch.from_numpy(gamma).float().to(device)  # no grad
    return gamma_torch  # treated as constant in backward


# ----------------------------
# One GD iteration on barycenter cost matrix Cb
# ----------------------------
def barycenter_step(Cb, lambda_vec, Cs_torch, ps_torch, p_b_torch, lr=1e-1):
    """
    Cb: (M,M), differentiable tensor
    lambda_vec: (K,), weights (part of graph)
    """
    K = len(Cs_torch)
    grad_Cb_total = torch.zeros_like(Cb)

    for k in range(K):
        Ck = Cs_torch[k]
        pk = ps_torch[k]
        gamma_k = gw_coupling_pot(Cb, Ck, p_b_torch, pk)  # constant (envelope)

        grad_Cb_k = gw_grad_Cb(Cb, Ck, gamma_k)
        grad_Cb_total = grad_Cb_total + lambda_vec[k] * grad_Cb_k

    # Gradient descent step on Cb (differentiable!)
    Cb = Cb - lr * grad_Cb_total
    Cb = 0.5 * (Cb + Cb.T)
    Cb.fill_diagonal_(0.0)

    return Cb


# ----------------------------
# Full Step 1: compute Z*(lambda) by unrolling
# ----------------------------
def compute_barycenter(lambda_vec, Cs_torch, ps_torch, p_b_torch,
                       M, n_iter=20, lr=1e-1):
    # Initialize barycenter cost matrix, e.g. random symmetric
    Cb = torch.randn(M, M, device=device) * 0.1
    Cb = 0.5 * (Cb + Cb.T)
    Cb.fill_diagonal_(0.0)
    Cb.requires_grad_(True)

    for _ in range(n_iter):
        Cb = barycenter_step(Cb, lambda_vec, Cs_torch, ps_torch, p_b_torch, lr=lr)

    return Cb  # this is C_b^*(lambda)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ---------------------------------------------------
    # Load all shapes
    # ---------------------------------------------------
    this_file = Path(__file__).resolve()
    data_path = this_file.parent / "simple_shapes"
    shape_files = ["square.png", "cross.png", "triangle.png", "star.png"]
    Xs = [load_binary_shape(data_path / fname) for fname in shape_files]
    ns = [len(X) for X in Xs]

    # ---------------------------------------------------
    # Compute cost matrices and uniform measures
    # ---------------------------------------------------
    Cs = [compute_cost_matrix(X) for X in Xs]
    ps = [ot.unif(n) for n in ns]

    # Example: remove shape i and use it as target
    i = 3
    Cs_except_i = [C for j, C in enumerate(Cs) if j != i]
    ps_except_i = [p for j, p in enumerate(ps) if j != i]
    C_target_np = Cs[i]      # target cost matrix
    p_target_np = ps[i]      # (not strictly needed here)

    # Barycenter support size
    M = 30
    p_b_np = ot.unif(M)

    # ----------------------------
    # Convert data to torch
    # ----------------------------
    Cs_torch = [torch.from_numpy(C).float().to(device) for C in Cs_except_i]
    ps_torch = [torch.from_numpy(p).float().to(device) for p in ps_except_i]
    C_target = torch.from_numpy(C_target_np).float().to(device)
    p_b_torch = torch.from_numpy(p_b_np).float().to(device)

    # lambda as a torch parameter (on simplex or not, up to you)
    lambda_init = torch.ones(len(Cs_torch), device=device) / len(Cs_torch)
    lambda_vec = lambda_init.clone().requires_grad_(True)

    # ----------------------------
    # Step 1: compute barycenter C_b^*(lambda)
    # ----------------------------
    C_b_star = compute_barycenter(
        lambda_vec, Cs_torch, ps_torch, p_b_torch,
        M=M, n_iter=20, lr=1e-1
    )

    # ----------------------------
    # Outer loss: GW between barycenter and target
    # ----------------------------
    loss = GWFunction.apply(C_b_star, C_target)

    # Backpropagate
    loss.backward()

    print("Loss:", loss.item())
    print("Gradient dPhi/dlambda:")
    print(lambda_vec.grad)
