"""Separable worst-case value and restoration allowance (Proposition 3, Table 2).

For a separable policy pi (repair probabilities), the separable worst-case value is
    V_(pi; P) = sum_i sum_s q_i(s) Y_1i(s),
    Y_ti(s) = sum_a pi_ti(a|s) [ r(s, a) + min_{p in P_i(s,a)} sum_s' p(s') Y_{t+1,i}(s') ],   Y_{T+1,i} = 0,
and the worst-case consumption C_t(pi; P) is the analogous recursion with maxima, the reward D(s, a) in period t and
zero rewards in all other periods. The restoration allowance Gamma(pi; P) is eq. (EC.allowance) of the electronic
companion; V_(pi; P) - Gamma(pi; P) bounds the worst-case value of the restored policy from below.
"""
from math import erfc, pi as PI, sqrt

import numpy as np

from .ambiguity import worst_case_kernel


def robust_policy_evaluation(prob, P_hat, rho, R, q, mask, sign=+1, D=None, t_target=None):
    """Robust evaluation of the separable policy with repair probabilities prob (T, I, S).

    sign=+1: separable worst-case value (minimum over the ambiguity set) of the total reward.
    sign=-1 with D (S, A) and t_target: worst-case (maximum) expected consumption sum_i E[D(S_t, A_t)] in period t_target.
    """
    T, I, S = prob.shape
    U = np.zeros((I, S))
    last = T - 1 if t_target is None else t_target
    for t in reversed(range(last + 1)):
        Pw = worst_case_kernel(P_hat, rho, sign * U, mask) if t < T - 1 or t_target is not None else P_hat
        cont = np.einsum("isaj,ij->isa", Pw, U) if (t < last) else np.zeros((I, S, 2))
        if t_target is None:
            stage = np.broadcast_to(R, (I, S, 2))
        else:
            stage = D[None, :, :] * (t == t_target) * np.ones((I, 1, 1))
        Q = stage + cont
        U = (1 - prob[t]) * Q[..., 0] + prob[t] * Q[..., 1]
    return float(np.sum(q * U))


def allowance(C_bar, budget, T, I, r_bar, D_max=1.0, D_min=1.0):
    """Restoration allowance Gamma = r_bar sum_t (T - t + 1) [psi_t(C_t - b_t) / D_min + varpi_t(C_t - b_t)].

    psi_t(e) = e_+ + min{D_max sqrt(t I) / 2, int_{(-e)_+}^inf exp(-2 w^2 / (t I D_max^2)) dw} and
    varpi_t(e) = min{1, exp(-2 (-e)_+^2 / (t I D_max^2))}, with 1-based t (single resource, N = 1).
    """
    tot = 0.0
    for t in range(T):
        u = C_bar[t] - budget
        It = I * (t + 1)   # variance proxy t I (1-based t)
        tail = D_max * sqrt(It) * sqrt(PI / 8) * erfc(sqrt(2) * max(-u, 0.0) / (D_max * sqrt(It)))
        g = max(u, 0.0) + min(0.5 * D_max * sqrt(It), tail)
        h = min(1.0, np.exp(-2 * max(-u, 0.0) ** 2 / (It * D_max ** 2)))
        tot += (T - t) * (g / D_min + h)          # (T - t + 1) with 1-based t
    return r_bar * tot


def certificate(prob, P_hat, rho, R, q, mask, budget):
    """Separable worst-case value, restoration allowance and worst-case consumptions (T,) of a separable policy."""
    T, I, S = prob.shape
    D = np.array([[0.0, 1.0]] * S)
    v_rob = robust_policy_evaluation(prob, P_hat, rho, R, q, mask)
    C_bar = np.array([robust_policy_evaluation(prob, P_hat, rho, R, q, mask, sign=-1, D=D, t_target=t) for t in range(T)])
    return v_rob, allowance(C_bar, budget, T, I, float(R.max())), C_bar
