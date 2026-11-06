"""Figure 1(b) and Table EC.1 (Example 1): the robust Lagrangian bound phi(mu) = L((mu, mu); P) is neither convex nor
concave.

Instance: I = 2 constituent MDPs, T = 2, states {1, 2}, actions {0, 1} (action 1 uses one unit of the resource),
b_1 = b_2 = 1, both start in state 1, l_inf balls of radius 0.1 around the center kernel
p_hat(1|1,1) = 0.6, p_hat(1|1,0) = 0.5, p_hat(1|2,1) = 0.5, p_hat(1|2,0) = 0.75. Since p_hat(1|s,a) lies in
[0.1, 0.9], min_{p in P(s,a)} p.w = p_hat.w - 0.1 |w(1) - w(2)|, and phi has the closed form of the electronic companion.
phi is piecewise linear; the script computes it in exact rational arithmetic, prints its breakpoints on [0, 1] (the
pgfplots coordinates of Figure 1(b)) and the value functions of Table EC.1, and optionally draws the curve.
Usage:  python experiments/fig1_nonconvex.py [--pdf figures/fig1_robust_lagrangian.pdf]
"""
import argparse
from fractions import Fraction as F

REWARDS = {1: {(1, 0): F(2, 10), (1, 1): F(3, 10), (2, 0): F(1, 10), (2, 1): F(8, 10)},
           2: {(1, 0): F(0), (1, 1): F(1), (2, 0): F(5, 10), (2, 1): F(6, 10)}}
P_STAY = {0: F(5, 10), 1: F(6, 10)}   # p_hat(1 | 1, a): probability of staying in state 1 from state 1
RADIUS = F(1, 10)


def value_functions(i, mu):
    """W_2(1), W_2(2), Q_1(1, 0), Q_1(1, 1), W_1(1) of constituent MDP i at the multiplier mu."""
    r = REWARDS[i]
    W2 = {s: max(r[(s, 0)], r[(s, 1)] - mu) for s in (1, 2)}
    d = abs(W2[1] - W2[2])
    Q0 = r[(1, 0)] + P_STAY[0] * W2[1] + (1 - P_STAY[0]) * W2[2] - RADIUS * d
    Q1 = r[(1, 1)] - mu + P_STAY[1] * W2[1] + (1 - P_STAY[1]) * W2[2] - RADIUS * d
    return W2[1], W2[2], Q0, Q1, max(Q0, Q1)


def phi(mu):
    """phi(mu) = sum_t mu b_t + W_11(1) + W_12(1) with b_1 = b_2 = 1."""
    return 2 * mu + sum(value_functions(i, mu)[-1] for i in (1, 2))


def breakpoints(n_grid=2200):
    """Kinks of the piecewise linear function phi on [0, 1], located exactly by intersecting adjacent pieces."""
    grid = [F(k, n_grid) for k in range(n_grid + 1)]
    vals = [phi(m) for m in grid]
    slopes = [(vals[k + 1] - vals[k]) * n_grid for k in range(n_grid)]
    pts = [(grid[0], vals[0])]
    for k in range(1, n_grid):
        if slopes[k] != slopes[k - 1]:
            # line through segment k-1 and line through segment k + 1 (segment k may contain the kink)
            a1, b1 = slopes[k - 1], vals[k - 1] - slopes[k - 1] * grid[k - 1]
            j = k + 1 if k + 1 < n_grid and slopes[k + 1] != slopes[k] else k
            a2, b2 = slopes[j], vals[j] - slopes[j] * grid[j]
            x = (b2 - b1) / (a1 - a2)
            if not pts or x != pts[-1][0]:
                pts.append((x, phi(x)))
    pts.append((grid[-1], vals[-1]))
    return pts


def fmt(v):
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", default=None, help="also draw the curve into this PDF file")
    a = p.parse_args()
    pts = breakpoints()
    print("pgfplots coordinates of Figure 1(b):")
    print("coordinates {" + " ".join(f"({fmt(x)},{fmt(y)})" for x, y in pts) + "};")
    print("\nTable EC.1:  mu | MDP 1: W2(1) W2(2) Q1(1,0) Q1(1,1) W1(1) | MDP 2: same | phi(mu)")
    for mu in (F(0), F(1, 10), F(2, 10), F(4, 10), F(7, 10)):
        row = [v for i in (1, 2) for v in value_functions(i, mu)]
        print(f"{fmt(mu):>4} | " + " ".join(f"{float(v):5.2f}" for v in row[:5]) + " | "
              + " ".join(f"{float(v):5.2f}" for v in row[5:]) + f" | {float(phi(mu)):.2f}")
    if a.pdf:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10})
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        ax.plot([float(x) for x, _ in pts], [float(y) for _, y in pts], color="#1f4e99", lw=1.4)
        ax.set_xlim(0, 1); ax.set_ylim(2.36, 2.58)
        ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$L(\mu;\mathcal{P})$"); ax.grid(alpha=0.3, lw=0.5)
        fig.tight_layout(); fig.savefig(a.pdf)
        print("saved", a.pdf)
