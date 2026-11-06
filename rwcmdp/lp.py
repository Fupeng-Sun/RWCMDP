"""Fluid linear programs (Sections 3 and 4), solved with HiGHS through scipy.optimize.linprog.

* FLP(p): the fluid LP of the WCMDP under a kernel p, which may be a kernel sequence (time-varying);
* CDR(P): the constant decision rule LP, i.e. the fluid LP under the lower envelope p_low with the scaled budget
  sum_{i,s} x_ti(s, repair) <= beta * sum_{i,s,a} x_ti(s, a).

The resource is the number of repairs: D(s, a) = 1{a = repair}, budget b_t = beta * I.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .instance import REPAIR


def solve_fluid_lp(P, R, q, b, horizon, ratio_budget=False, flow_inequality=False):
    """Fluid LP for a kernel P of shape (I, S, A, S') or a kernel sequence of shape (T-1, I, S, A, S').

    Variables x[t, i, s, a] >= 0 (occupancy measures). Constraints: sum_a x_1i(s, a) = q_i(s); the flow constraints
    sum_a x_{t+1,i}(s', a) = sum_{s,a} p_ti(s'|s,a) x_ti(s, a) (<= with flow_inequality); and the budget, either
    sum_{i,s} x_ti(s, 1) <= b I or, with ratio_budget, sum_{i,s} x_ti(s, 1) <= b sum_{i,s,a} x_ti(s, a).
    The objective is sum_{t,i,s,a} r(s, a) x_ti(s, a).

    Returns the optimal value, x of shape (T, I, S, A) and the budget multipliers mu_t >= 0 (shape (T,)).
    If HiGHS fails, the LP is retried with the interior point and dual simplex methods and, as a last resort, with a
    tiny perturbation of the objective that breaks the degeneracy of identical machines.
    """
    n_machines = P.shape[-4]
    S, A = R.shape
    T, I = horizon, n_machines
    n_var = T * I * S * A
    var = np.arange(n_var).reshape(T, I, S, A)
    cost = -np.broadcast_to(R, (T, I, S, A)).ravel()

    # initial distribution: one row per (i, s)
    init_rows = np.broadcast_to(np.arange(I * S).reshape(I, S)[:, :, None], (I, S, A))
    A_init = sp.csr_matrix((np.ones(I * S * A), (init_rows.ravel(), var[0].ravel())), shape=(I * S, n_var))

    # flow: sum_a x_{t+1,i}(s', a) - sum_{s,a} x_ti(s, a) p_ti(s'|s, a), one row per (t, i, s')
    row_id = np.arange((T - 1) * I * S).reshape(T - 1, I, S)
    r_next = np.broadcast_to(row_id[:, :, :, None], (T - 1, I, S, A))
    r_prev = np.broadcast_to(row_id[:, :, None, None, :], (T - 1, I, S, A, S))  # (t, i, s, a, s')
    c_prev = np.broadcast_to(var[:-1, :, :, :, None], (T - 1, I, S, A, S))
    v_prev = -np.broadcast_to(P, (T - 1, I, S, A, S))
    A_flow = sp.csr_matrix(
        (np.concatenate([np.ones(r_next.size), v_prev.ravel()]),
         (np.concatenate([r_next.ravel(), r_prev.ravel()]), np.concatenate([var[1:].ravel(), c_prev.ravel()]))),
        shape=((T - 1) * I * S, n_var))

    # budget: one row per t
    budget_coef = np.zeros((I, S, A))
    budget_coef[:, :, REPAIR] = 1.0
    if ratio_budget:
        budget_coef -= b
    t_rows = np.broadcast_to(np.arange(T)[:, None, None, None], (T, I, S, A))
    A_budget = sp.csr_matrix((np.broadcast_to(budget_coef, (T, I, S, A)).ravel(), (t_rows.ravel(), var.ravel())),
                             shape=(T, n_var))
    b_budget = np.zeros(T) if ratio_budget else np.full(T, b * I)

    if flow_inequality:
        A_eq, b_eq = A_init, q.ravel()
        A_ub, b_ub = sp.vstack([A_flow, A_budget]), np.concatenate([np.zeros((T - 1) * I * S), b_budget])
    else:
        A_eq, b_eq = sp.vstack([A_init, A_flow]), np.concatenate([q.ravel(), np.zeros((T - 1) * I * S)])
        A_ub, b_ub = A_budget, b_budget
    pert = np.random.default_rng(12345).random(cost.size) * 1e-7   # fallbacks only: break the degeneracy of identical machines
    pert2 = np.random.default_rng(777).random(cost.size) * 1e-6
    attempts = [("highs", {}, cost), ("highs-ipm", {}, cost), ("highs-ds", {}, cost * (1 + pert)),
                ("highs", {"presolve": False}, cost * (1 + pert)), ("highs-ipm", {"presolve": False}, cost * (1 + pert2)),
                ("highs-ds", {"presolve": False}, 1000 * cost * (1 + pert2))]
    for method, options, c_ in attempts:
        options = {**options, "time_limit": 150.0}  # a pathological LP must not stall the whole experiment
        res = linprog(c_, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=method, options=options)
        if res.status == 0:
            break
    else:
        raise RuntimeError(f"LP solver failed: {res.message}")
    return -res.fun, res.x.reshape(T, I, S, A), -res.ineqlin.marginals[-T:]


def solve_cdr_lp(P_low, R, q, b, horizon, flow_inequality=True):
    """CDR LP: the fluid LP under the lower envelope P_low with the scaled budget beta * sum_{i,s,a} x_ti(s, a).

    Its optimal value is the CDR bound L_CDR(P). The experiments of Figures 3 and 4 state the flow constraints as
    inequalities; Table 2 states them as equalities, as in the definition of the CDR policy (Section 3.3). Since all
    rewards are nonnegative, the two versions have the same optimal value.
    """
    return solve_fluid_lp(P_low, R, q, b, horizon, ratio_budget=True, flow_inequality=flow_inequality)

