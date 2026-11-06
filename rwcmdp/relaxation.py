"""Robust Lagrangian relaxation and the subgradient method (Section 3.2, Algorithm 1).

For multipliers mu = (mu_1, ..., mu_T) >= 0 of the budget constraints, the robust Lagrangian bound decomposes into one
robust MDP per machine:

    W_ti(s) = max_a { r(s, a) - mu_t D(s, a) + min_{p in P_i(s,a)} sum_s' p(s') W_{t+1,i}(s') },   W_{T+1,i} = 0,
    L(mu; P) = sum_t mu_t b_t + sum_i sum_s q_i(s) W_1i(s),

with D(s, a) = 1{a = repair} and b_t = beta * I. The minimizing kernels form the kernel sequence p^mu, and
b_t - (expected consumption in period t under the maximizing actions and p^mu) is a subgradient of L(.; P) at mu.
"""
import numpy as np

from .ambiguity import worst_case_kernel


def robust_lagrangian(mu, P_hat, rho, R, b, mask, q, horizon):
    """Value L(mu; P), subgradient with respect to mu (shape (T,)) and Q-values (T, I, S, A) of the robust recursion.

    The inner minimization in the last period is irrelevant (W_{T+1} = 0) and uses p_hat. Ties in the maximization
    over actions go to `do nothing` (np.argmax returns the first maximizer).
    """
    T, I, S = horizon, q.shape[0], P_hat.shape[1]
    W = np.zeros((T + 1, I, S))
    dW = np.zeros((T + 1, I, S, T))
    Q = np.zeros((T, I, S, 2))
    for t in reversed(range(T)):
        Pw = P_hat if t == T - 1 else worst_case_kernel(P_hat, rho, W[t + 1], mask)
        Q[t] = R - mu[t] * np.array([0.0, 1.0]) + np.einsum("isaj,ij->isa", Pw, W[t + 1])
        a = np.argmax(Q[t], axis=-1)
        W[t] = np.take_along_axis(Q[t], a[..., None], -1)[..., 0]
        p = np.take_along_axis(Pw, a[..., None, None], 2)[:, :, 0]
        dW[t] = np.einsum("isj,ijk->isk", p, dW[t + 1])
        dW[t][:, :, t] -= (a == 1)
    value = b * I * mu.sum() + np.sum(q * W[0])
    grad = b * I + np.einsum("is,isk->k", q, dW[0])
    return value, grad, Q


def subgradient_method(P_hat, rho, mu0, R, b, mask, q, horizon, n_iter=150, eta=0.01):
    """Algorithm 1: projected subgradient steps mu <- max(mu - eta / sqrt(j) * g, 0), j = 1, ..., n_iter.

    Starts from mu0 (in the experiments: the budget multipliers of FLP(p_hat), the nominal optimum) and returns the
    iterate with the smallest bound as (L(mu~; P), mu~, Q-values at mu~).
    """
    mu = np.maximum(mu0, 0)
    best = None
    for j in range(1, n_iter + 1):
        v, g, Q = robust_lagrangian(mu, P_hat, rho, R, b, mask, q, horizon)
        if best is None or v < best[0]:
            best = (v, mu.copy(), Q)
        mu = np.maximum(mu - eta / np.sqrt(j) * g, 0)
    return best


def worst_case_kernel_sequence(P_hat, rho, mu, R, mask, horizon):
    """Kernel sequence p~ in P that attains the minima of the robust recursion at mu.

    Returns an array of shape (T-1, I, S, A, S'): entry t moves the machines from period t to period t+1.
    """
    T, I, S = horizon, P_hat.shape[0], P_hat.shape[1]
    W = np.zeros((T + 1, I, S))
    Pt = np.zeros((T - 1,) + P_hat.shape)
    for t in reversed(range(T)):
        Pw = P_hat if t == T - 1 else worst_case_kernel(P_hat, rho, W[t + 1], mask)
        if t < T - 1:
            Pt[t] = Pw
        Q = R - mu[t] * np.array([0.0, 1.0]) + np.einsum("isaj,ij->isa", Pw, W[t + 1])
        W[t] = Q.max(-1)
    return Pt
