import numpy as np
import ot
from ot.gromov import gromov_barycenters, gromov_wasserstein2


# -------------------------------------------------------------------
# Helpers: simplex projection and softmax mirror map (optional)
# -------------------------------------------------------------------

def project_onto_simplex(v):
    """
    Project a vector v in R^S onto the probability simplex:
        Δ = { x >= 0, sum(x) = 1 }
    Using the algorithm of Duchi et al.
    """
    v = np.asarray(v, dtype=float)
    if np.all(v >= 0) and np.isclose(v.sum(), 1.0):
        return v

    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w


def softmax(theta):
    """Softmax map for mirror descent parameterization λ = softmax(θ)."""
    theta = np.asarray(theta, dtype=float)
    theta = theta - np.max(theta)  # numerical stability
    e = np.exp(theta)
    return e / e.sum()


# -------------------------------------------------------------------
# Objective: λ ↦ GW^2(Y, Y_λ)
# -------------------------------------------------------------------

def gw_barycenter_loss(
    lam,
    Cs_templates, ps_templates,
    C_target, p_target,
    n_bary_samples=None,
    loss_fun='square_loss',
    max_iter_bary=50,
    max_iter_gw=50,
):
    """
    lam:          array of shape (S,), point on simplex (weights for templates)
    Cs_templates: list/array of S cost matrices C_s (each (n_s, n_s) or same size)
    ps_templates: list/array of S histograms p_s (each length n_s, sum to 1)
    C_target:     cost matrix for target Y (shape (n_y, n_y))
    p_target:     histogram for target Y (length n_y)
    returns:      GW^2(Y, Y_λ)
    """

    S = len(Cs_templates)
    lam = np.asarray(lam, dtype=float)
    assert lam.shape == (S,)
    assert np.all(lam >= -1e-12), "λ must be nonnegative (before projection)."
    lam = lam / lam.sum()  # make sure exactly on simplex

    # Number of points in barycenter: if not given, use size of first template
    if n_bary_samples is None:
        n_bary_samples = Cs_templates[0].shape[0]

    # Weights of barycenter support (can fix uniform or reuse target weights etc.)
    p_bary = np.ones(n_bary_samples) / n_bary_samples

    # POT's GW barycenter (synthetic Y_λ)
    C_bar = gromov_barycenters(
        n_bary_samples,
        Cs_templates,
        ps_templates,
        lam,
        p_b=p_bary,
        loss_fun=loss_fun,
        max_iter=max_iter_bary,
        tol=1e-9,
        verbose=False,
    )
    # Now compute GW^2 between target Y and synthetic barycenter Y_λ
    gw2 = gromov_wasserstein2(
        C_target, C_bar,
        p_target, p_bary,
        loss_fun=loss_fun,
        max_iter=max_iter_gw,
        tol=1e-9,
        verbose=False,
    )
    return gw2


# -------------------------------------------------------------------
# Finite-difference gradient (replace by autodiff if you reimplement POT in JAX/torch)
# -------------------------------------------------------------------

def numerical_grad(func, lam, eps=1e-4, **kwargs):
    """
    Simple forward finite-difference gradient of func at lam.
    func(lam, **kwargs) -> scalar
    """
    lam = np.asarray(lam, dtype=float)
    grad = np.zeros_like(lam)
    f0 = func(lam, **kwargs)
    for i in range(lam.size):
        lam_pert = lam.copy()
        lam_pert[i] += eps
        lam_pert = project_onto_simplex(lam_pert)
        f_eps = func(lam_pert, **kwargs)
        grad[i] = (f_eps - f0) / eps
    return grad


# -------------------------------------------------------------------
# Projected gradient descent on the simplex
# -------------------------------------------------------------------

def optimize_lambda_projected_gd(
    Cs_templates, ps_templates,
    C_target, p_target,
    S=None,
    n_bary_samples=None,
    lr=1e-1,
    n_iter=200,
    lam_init=None,
    verbose_every=10,
):
    """
    Projected gradient descent:
        λ^{k+1} = Π_Δ( λ^k - lr * ∇_λ GW^2(Y, Y_λ) )
    """

    if S is None:
        S = len(Cs_templates)
    assert S == len(Cs_templates) == len(ps_templates)

    if lam_init is None:
        lam = np.ones(S) / S
    else:
        lam = project_onto_simplex(lam_init)

    history = []

    for it in range(n_iter):
        # Compute gradient numerically
        grad_lam = numerical_grad(
            gw_barycenter_loss,
            lam,
            Cs_templates=Cs_templates,
            ps_templates=ps_templates,
            C_target=C_target,
            p_target=p_target,
            n_bary_samples=n_bary_samples,
        )

        # Gradient step + projection
        lam = lam - lr * grad_lam
        lam = project_onto_simplex(lam)

        # Track loss
        loss_val = gw_barycenter_loss(
            lam,
            Cs_templates, ps_templates,
            C_target, p_target,
            n_bary_samples=n_bary_samples,
        )
        history.append(loss_val)

        if verbose_every is not None and it % verbose_every == 0:
            print(f"Iter {it:4d} | loss = {loss_val:.6e} | λ = {lam}")

    return lam, history


# -------------------------------------------------------------------
# OPTIONAL: Mirror descent via softmax (unconstrained θ ∈ R^S)
# -------------------------------------------------------------------

def optimize_lambda_mirror_descent(
    Cs_templates, ps_templates,
    C_target, p_target,
    S=None,
    n_bary_samples=None,
    lr=1e-1,
    n_iter=200,
    theta_init=None,
    verbose_every=10,
):
    """
    Mirror descent with softmax parameterization:
        λ = softmax(θ), θ ∈ R^S unconstrained
    We still use finite-difference on λ here; in a JAX/torch implementation
    you would differentiate wrt θ directly with autodiff.
    """
    if S is None:
        S = len(Cs_templates)
    assert S == len(Cs_templates) == len(ps_templates)

    if theta_init is None:
        theta = np.zeros(S)
    else:
        theta = np.asarray(theta_init, dtype=float)

    history = []

    for it in range(n_iter):
        lam = softmax(theta)

        # gradient wrt λ
        grad_lam = numerical_grad(
            gw_barycenter_loss,
            lam,
            Cs_templates=Cs_templates,
            ps_templates=ps_templates,
            C_target=C_target,
            p_target=p_target,
            n_bary_samples=n_bary_samples,
        )

        # chain rule: dθ = J_softmax(θ)^T * ∇_λ f
        # J_softmax(θ)_{ij} = λ_i (δ_ij - λ_j)
        # => ∂f/∂θ_k = sum_i ∂f/∂λ_i * ∂λ_i/∂θ_k
        #            = sum_i g_i * λ_i (δ_{ik} - λ_k)
        #            = λ_k (g_k - ⟨g, λ⟩)
        g_dot_lam = float(np.dot(grad_lam, lam))
        grad_theta = lam * (grad_lam - g_dot_lam)

        # gradient step in θ
        theta = theta - lr * grad_theta

        # record loss
        loss_val = gw_barycenter_loss(
            lam,
            Cs_templates, ps_templates,
            C_target, p_target,
            n_bary_samples=n_bary_samples,
        )
        history.append(loss_val)

        if verbose_every is not None and it % verbose_every == 0:
            print(f"Iter {it:4d} | loss = {loss_val:.6e} | λ = {lam}")

    lam_final = softmax(theta)
    return lam_final, history


# -------------------------------------------------------------------
# Example usage (you plug in your own data here)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # === TODO: Replace this toy data by your actual templates and target ===

    S = 3  # number of templates
    n_s = 10  # template size
    n_y = 12  # target size

    rng = np.random.default_rng(0)

    # Random symmetric cost matrices for templates
    Cs_templates = []
    ps_templates = []
    for s in range(S):
        X = rng.normal(size=(n_s, 2))   # random points in R^2
        C = ot.dist(X, X, metric='euclidean')  # pairwise distances
        Cs_templates.append(C)
        ps_templates.append(np.ones(n_s) / n_s)

    # Target structure Y
    Y = rng.normal(size=(n_y, 2))
    C_target = ot.dist(Y, Y, metric='euclidean')
    p_target = np.ones(n_y) / n_y

    # Run projected GD
    lam_opt, history = optimize_lambda_projected_gd(
        Cs_templates, ps_templates,
        C_target, p_target,
        S=S,
        n_bary_samples=15,
        lr=0.5,
        n_iter=50,
        verbose_every=5,
    )

    print("\nOptimal λ (projected GD) =", lam_opt)

    # Or mirror descent (+ softmax parameterization)
    lam_opt_md, history_md = optimize_lambda_mirror_descent(
        Cs_templates, ps_templates,
        C_target, p_target,
        S=S,
        n_bary_samples=15,
        lr=0.5,
        n_iter=50,
        verbose_every=5,
    )

    print("\nOptimal λ (mirror descent) =", lam_opt_md)
