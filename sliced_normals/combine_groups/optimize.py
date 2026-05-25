import hashlib
import autograd.numpy as np
from autograd.numpy.linalg import inv, cholesky
from pymanopt import Problem
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers import SteepestDescent
import pymanopt.function
from autograd.scipy.special import logsumexp

def contraction(U):
    """
    Smooth map U -> C(U) with ||C(U)||_2 < 1 (via Frobenius norm bound).
    """
    frob = np.sqrt(np.sum(U ** 2)+1e-5)
    scale = 1.0 / (1.0 + frob)   # in (0,1]
    return scale * U             # ||C||_F < 1 => ||C||_2 < 1

def estimate_optimal_P12_reparam(
    Y1, Y2, Y1g, Y2g, volume,
    P11, P22,
    verbosity=2,
    max_iterations=1000,
    min_gradient_norm=1e-4,
    winsor_threshold=1e-10,
    return_details=False,
    n_restarts=1,              # number of additional random restarts after default run
    random_state=None          # reproducibility for random restarts
):
    """
    Final objective (Eq. 21) + winsorization, with multi-start optimization.

    Optimization schedule:
      1) Run once with the default initialization U0 = 0 (original behavior)
      2) Run `n_restarts` additional optimizations with U0 ~ N(0,1) entrywise
      3) Return the best solution (lowest objective value among finite runs)

    Objective:
        min_U  (1/n) * sum_i [ y1_i^T P12(U) y2_i ]
             + log( (volume/m) * sum_j w_j * exp(-2 y1g_j^T P12(U) y2g_j) )

    with
        w_j = exp( - y1g_j^T P11 y1g_j - y2g_j^T P22 y2g_j )

    and winsorized log-integral:
        log_integral <- max(log_integral, log(winsor_threshold))
    """
    n, d1 = Y1.shape
    n2, d2 = Y2.shape
    m, d1g = Y1g.shape
    m2, d2g = Y2g.shape

    if n2 != n:
        raise ValueError("Y1 and Y2 must have the same number of rows.")
    if m2 != m:
        raise ValueError("Y1g and Y2g must have the same number of rows.")
    if d1g != d1 or d2g != d2:
        raise ValueError("Grid feature dimensions must match data feature dimensions.")
    if winsor_threshold <= 0:
        raise ValueError("winsor_threshold must be positive.")
    if volume <= 0:
        raise ValueError("volume must be positive.")
    if n_restarts < 0:
        raise ValueError("n_restarts must be >= 0.")

    L1 = cholesky(P11)
    L2 = cholesky(P22)

    # log w_j = - y1g^T P11 y1g - y2g^T P22 y2g
    q11_g = np.sum((Y1g @ P11) * Y1g, axis=1)
    q22_g = np.sum((Y2g @ P22) * Y2g, axis=1)
    log_w = -(q11_g + q22_g)

    manifold = Euclidean(d1, d2)

    @pymanopt.function.autograd(manifold)
    def cost(U):
        C = contraction(U)
        P12 = L1 @ C @ L2

        # Data term: (1/n) sum y1_i^T P12 y2_i   <-- no factor 2
        cross_data = np.sum((Y1 @ P12) * Y2, axis=1)
        term1 = np.mean(cross_data)

        # Log integral term: log( (volume/m) sum_j w_j exp(-2 cross_grid_j) )
        cross_grid = np.sum((Y1g @ P12) * Y2g, axis=1)
        raw_log_int = logsumexp(log_w - 2.0 * cross_grid) - np.log(m) + np.log(volume)

        # Winsorization
        log_integral = np.maximum(raw_log_int, np.log(winsor_threshold))

        return term1 + log_integral

    def _run_single(U0, run_verbosity):
        problem = Problem(manifold=manifold, cost=cost)
        optimizer = SteepestDescent(
            verbosity=run_verbosity,
            max_iterations=max_iterations,
            min_gradient_norm=min_gradient_norm
        )
        result = optimizer.run(problem, initial_point=U0)

        U_opt = result.point
        C_opt = contraction(U_opt)
        P12_opt = L1 @ C_opt @ L2

        cross_data_opt = np.sum((Y1 @ P12_opt) * Y2, axis=1)
        term1_opt = np.mean(cross_data_opt)

        cross_grid_opt = np.sum((Y1g @ P12_opt) * Y2g, axis=1)
        raw_log_int_opt = logsumexp(log_w - 2.0 * cross_grid_opt) - np.log(m) + np.log(volume)
        log_integral_opt = np.maximum(raw_log_int_opt, np.log(winsor_threshold))

        obj_opt = term1_opt + log_integral_opt

        details = {
            "U_opt": U_opt,
            "C_opt": C_opt,
            "term1_opt": term1_opt,
            "raw_log_integral_opt": raw_log_int_opt,
            "winsorized_log_integral_opt": log_integral_opt,
            "objective_opt": obj_opt,
            "result": result,
        }
        return P12_opt, details

    rng = np.random.default_rng(random_state)

    # Build initialization list:
    #   first run = default zero init (original behavior),
    #   remaining runs = random Normal(0,1) in U0
    init_list = [np.zeros((d1, d2))]
    for _ in range(n_restarts):
        init_list.append(5*rng.standard_normal((d1, d2)))

    best_P12 = None
    best_details = None
    best_obj = np.inf
    restart_summaries = []

    for r, U0 in enumerate(init_list):
        # Keep original verbosity on the first/default run; silence restarts unless verbosity>=3
        run_verbosity = verbosity if (r == 0 or verbosity >= 3) else 0
        init_type = "default_zero" if r == 0 else "random_normal"

        try:
            P12_r, details_r = _run_single(U0, run_verbosity)
            obj_r = details_r["objective_opt"]
            print("Restart ", r, "/", n_restarts," Obj: ", obj_r)

            finite_flag = bool(np.isfinite(obj_r))
            restart_summaries.append({
                "restart_index": r,
                "init_type": init_type,
                "init_fro_norm": float(np.sqrt(np.sum(U0 ** 2))),
                "objective_opt": float(obj_r) if np.isfinite(obj_r) else np.nan,
                "is_finite": finite_flag,
            })

            if finite_flag and obj_r < best_obj:
                best_obj = obj_r
                best_P12 = P12_r
                best_details = details_r

        except Exception as e:
            restart_summaries.append({
                "restart_index": r,
                "init_type": init_type,
                "init_fro_norm": float(np.sqrt(np.sum(U0 ** 2))),
                "objective_opt": np.nan,
                "is_finite": False,
                "error": str(e),
            })

    if best_P12 is None:
        raise RuntimeError(
            "All optimization runs failed or returned non-finite objectives. "
            "This often happens if contraction(U) is non-differentiable at U=0 (autograd NaNs)."
        )

    if not return_details:
        return best_P12

    # Attach multi-start diagnostics while preserving original detail keys
    best_details = dict(best_details)
    best_details["n_restarts_random"] = int(n_restarts)
    best_details["n_total_runs"] = int(len(init_list))
    best_details["best_objective_opt"] = float(best_obj)
    best_details["restart_summaries"] = restart_summaries

    return best_P12, best_details





# -----------------------------------------------------------
# 3.  High-level wrapper that builds the full B-matrix
#     (updated for the NEW centered formulation)
# -----------------------------------------------------------
def combine_B_blocks_reparam(B1, B2,
                             data1, data2,
                             degree=2, n_grid=10000,
                             verbosity=2, max_iterations=1000,
                             min_gradient_norm=1e-4,
                             winsor_threshold=1e-10,
                             n_restarts=1,
                             return_details=False):
    """
    Combines two fitted block matrices B1, B2 by optimizing the cross-block P12
    using the NEW reparameterized objective (centered formulation).

    Returns
    -------
    mu : ndarray
        Combined mean vector [mu1, mu2].
    P : ndarray
        Combined precision matrix [[P11, P12], [P12^T, P22]].
    (optional) details : dict
        Diagnostics from estimate_optimal_P12_reparam if return_details=True.
    """

    # ----- 3.1  Extract block pieces from the two FMLE blocks -----
    p1 = B1.shape[0] - 1
    p2 = B2.shape[0] - 1

    # Precision blocks from B
    P11 = 2.0 * B1[1:1 + p1, 1:1 + p1]
    P22 = 2.0 * B2[1:1 + p2, 1:1 + p2]

    P11_inv = inv(P11)
    P22_inv = inv(P22)

    # Means implied by B blocks
    mu1 = -2.0 * B1[0, 1:1 + p1] @ P11_inv
    mu2 = -2.0 * B2[0, 1:1 + p2] @ P22_inv
    mu = np.concatenate([mu1, mu2])

    # ----- 3.2  Feature matrices & integration grids -------------
    if not isinstance(data1, pd.DataFrame):
        data1 = pd.DataFrame(data1, columns=[f"X1_{i}" for i in range(data1.shape[1])])
    if not isinstance(data2, pd.DataFrame):
        data2 = pd.DataFrame(data2, columns=[f"X2_{i}" for i in range(data2.shape[1])])

    Z1_df, Z1g_df, vol1 = get_F_and_Random_Samples(data1, degree, n_grid)
    Z2_df, Z2g_df, vol2 = get_F_and_Random_Samples(data2, degree, n_grid)

    # Keep feature columns only (drop intercept column)
    Z1  = np.asarray(Z1_df.iloc[:, 1:], dtype=np.float64)
    Z2  = np.asarray(Z2_df.iloc[:, 1:], dtype=np.float64)
    Z1g = np.asarray(Z1g_df.iloc[:, 1:], dtype=np.float64)
    Z2g = np.asarray(Z2g_df.iloc[:, 1:], dtype=np.float64)

    volume = float(vol1 * vol2)

    # ----- 3.3  Center features for the NEW formulation ----------
    # Optimizer now expects Y = Z - mu
    if Z1.shape[1] != mu1.shape[0]:
        raise ValueError(
            f"Dimension mismatch: feature dim for block 1 is {Z1.shape[1]}, "
            f"but mu1 has length {mu1.shape[0]}."
        )
    if Z2.shape[1] != mu2.shape[0]:
        raise ValueError(
            f"Dimension mismatch: feature dim for block 2 is {Z2.shape[1]}, "
            f"but mu2 has length {mu2.shape[0]}."
        )

    Y1  = Z1  - mu1
    Y2  = Z2  - mu2
    Y1g = Z1g - mu1
    Y2g = Z2g - mu2

    # ----- 3.4  Optimise U  ->  get P12 --------------------------
    if return_details:
        P12_opt, opt_details = estimate_optimal_P12_reparam(
            Y1, Y2, Y1g, Y2g, volume,
            P11, P22,
            verbosity=verbosity,
            max_iterations=max_iterations,
            min_gradient_norm=min_gradient_norm,
            winsor_threshold=winsor_threshold,
            n_restarts=n_restarts,
            return_details=True
        )
    else:
        P12_opt = estimate_optimal_P12_reparam(
            Y1, Y2, Y1g, Y2g, volume,
            P11, P22,
            verbosity=verbosity,
            max_iterations=max_iterations,
            min_gradient_norm=min_gradient_norm,
            winsor_threshold=winsor_threshold,
            n_restarts=n_restarts,
            return_details=False
        )

    # ----- 3.5  Assemble the full precision block ----------------
    P = np.block([
        [P11,       P12_opt],
        [P12_opt.T, P22    ]
    ])

    if return_details:
        details = {
            "mu1": mu1,
            "mu2": mu2,
            "mu": mu,
            "P11": P11,
            "P22": P22,
            "P12_opt": P12_opt,
            "volume": volume,
            "vol1": float(vol1),
            "vol2": float(vol2),
            "opt_details": opt_details
        }
        return mu, P, details

    return mu, P

def combine_B_blocks_indep(B1, B2,
                             data1, data2,
                             degree=2, n_grid=10000,
                             verbosity=2, max_iterations=1000,
                             min_gradient_norm=1e-4,
                             winsor_threshold=1e-10):

    # ----- 3.1  Extract block pieces from the two FMLE blocks -----
    p1 = B1.shape[0] - 1
    p2 = B2.shape[0] - 1

    P11 = 2 * B1[1:1 + p1, 1:1 + p1]
    P22 = 2 * B2[1:1 + p2, 1:1 + p2]
    P11_inv = inv(P11)
    P22_inv = inv(P22)

    mu1 = -2 * B1[0, 1:1 + p1] @ P11_inv
    mu2 = -2 * B2[0, 1:1 + p2] @ P22_inv
    mu  = np.concatenate([mu1, mu2])

    # ----- 3.2  Feature matrices & integration grids -------------
    if not isinstance(data1, pd.DataFrame):
        data1 = pd.DataFrame(data1, columns=[f"X1_{i}" for i in range(data1.shape[1])])
    if not isinstance(data2, pd.DataFrame):
        data2 = pd.DataFrame(data2, columns=[f"X2_{i}" for i in range(data2.shape[1])])

    Z1_df, _, _ = get_F_and_Random_Samples(data1, degree, n_grid)
    Z2_df, _, _ = get_F_and_Random_Samples(data2, degree, n_grid)

    Z1   = np.asarray(Z1_df.iloc[:, 1:], dtype=np.float64)
    Z2   = np.asarray(Z2_df.iloc[:, 1:], dtype=np.float64)

    _, d1 = Z1.shape
    _, d2 = Z2.shape
    P12_opt = np.zeros((d1,d2))
    # ----- 3.4  Assemble the full precision block ----------------
    P = np.block([[P11,        P12_opt],
                  [P12_opt.T,  P22     ]])

    return mu, P