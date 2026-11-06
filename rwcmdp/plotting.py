"""Figures 3 and 4 of Section 5 (one PDF per panel plus one legend strip per figure).

Every panel is drawn at its printed size with a fixed axes box, so that the panels of a figure line up.
Figure 3: median over the histories and 25%-75% band of the policy ratio against the radius multiplier c, with a
broken y-axis (the random policy lies near 0.63). Figure 4: mean policy ratio with the cross-validated multiplier
against the repair ratio beta (K = 5, histories 0-4) and against K (beta = 0.25, histories 0-9).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 9, "mathtext.fontset": "cm"})
POLICIES = ["mle", "sub", "cdr", "random"]
COLOR = {"mle": "#555555", "sub": "#1f77b4", "cdr": "#6a3d9a", "random": "#2ca58d"}
LABEL = {"mle": "MLE", "sub": "Subgradient", "cdr": "CDR", "random": "Random"}
MARKER = {"mle": "s", "sub": "o", "cdr": "^", "random": "d"}
LW_MED, LW_Q = 0.7, 0.35           # Figure 3: median and quartile line widths (pt)
PANEL = (3.18, 2.3)                # inches; panels are included at 0.49 \textwidth
ADJ = dict(left=0.175, right=0.975, bottom=0.19, top=0.975)
FIG3_HEIGHT = 2.0
FIG4_BETA_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7)
FIG4_BETA_K = 0.25


def _save(fig, out_dir, name, png=False):
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    if png:
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=200)
    plt.close(fig)


def _legend_strip(out_dir, name, handles, png=False):
    fig = plt.figure(figsize=(6.5, 0.24))
    fig.legend(handles, [LABEL[n] for n in POLICIES], loc="center", ncol=4, frameon=False, handlelength=2.2,
               columnspacing=2.5, borderaxespad=0)
    _save(fig, out_dir, name, png)


def _iqr_band(ax, x, M, color):
    """M: array (histories, len(x)); median line, dashed quartile lines and 25-75% band."""
    q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
    ax.plot(x, med, color=color, lw=LW_MED)
    ax.plot(x, q1, color=color, lw=LW_Q, ls="--")
    ax.plot(x, q3, color=color, lw=LW_Q, ls="--")
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)


def _flat_band(ax, x, v, color):
    """Radius-independent policy: horizontal median line and band."""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([x[0], x[-1]], [med, med], color=color, lw=LW_MED)
    ax.plot([x[0], x[-1]], [q1, q1], color=color, lw=LW_Q, ls="--")
    ax.plot([x[0], x[-1]], [q3, q3], color=color, lw=LW_Q, ls="--")
    ax.fill_between([x[0], x[-1]], [q1, q1], [q3, q3], color=color, alpha=0.18, lw=0)


def fig3_data(fig3, K, c_max=5):
    """Radii and arrays (histories, radii) of the policy ratios of Figure 3 for one K."""
    items = [d for (k, _), d in fig3.items() if k == K]
    cs = sorted(set.intersection(*[set(d["cdr"]) & set(d["sub"]) for d in items]))
    cs = [c for c in cs if c <= c_max]
    return dict(c=np.array(cs, dtype=float),
                mle=np.array([d["mle"] for d in items]), random=np.array([d["random"] for d in items]),
                sub=np.array([[d["sub"][c] for c in cs] for d in items]),
                cdr=np.array([[d["cdr"][c] for c in cs] for d in items]))


def figure3(fig3, out_dir, png=False):
    """policy_ratio_K{K}.pdf for every K and legend_radius.pdf."""
    for K in sorted({k for k, _ in fig3}):
        d = fig3_data(fig3, K)
        x = d["c"]
        fig = plt.figure(figsize=(PANEL[0], FIG3_HEIGHT))
        l, r_ = ADJ["left"], ADJ["right"]
        b0, t0 = ADJ["bottom"] * PANEL[1] / FIG3_HEIGHT, 1 - (1 - ADJ["top"]) * PANEL[1] / FIG3_HEIGHT
        gap, low = 0.035 * PANEL[1] / FIG3_HEIGHT, 0.15 * PANEL[1] / FIG3_HEIGHT
        ax_lo = fig.add_axes([l, b0, r_ - l, low])
        ax = fig.add_axes([l, b0 + low + gap, r_ - l, t0 - b0 - low - gap], sharex=ax_lo)
        for a in (ax, ax_lo):
            _flat_band(a, x, d["mle"], COLOR["mle"])
            for name in ("sub", "cdr"):
                _iqr_band(a, x, d[name], COLOR[name])
            _flat_band(a, x, d["random"], COLOR["random"])
            a.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0.78, 1.006); ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        ax_lo.set_ylim(0.61, 0.65); ax_lo.set_yticks([0.63])
        ax.spines["bottom"].set_visible(False); ax_lo.spines["top"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none", color="k", mec="k", mew=0.8, clip_on=False)
        ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kw)
        ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)
        ax_lo.set_xlabel(r"radius multiplier $c$"); ax_lo.set_xlim(x[0] - 0.05, x[-1] + 0.05)
        fig.text(0.005, (b0 + t0) / 2, "Policy ratio", rotation=90, va="center", ha="left",
                 fontsize=plt.rcParams["axes.labelsize"])
        _save(fig, out_dir, f"policy_ratio_K{K}", png)
    _legend_strip(out_dir, "legend_radius",
                  [(Patch(facecolor=COLOR[n], alpha=0.18, lw=0), Line2D([], [], color=COLOR[n], lw=LW_MED)) for n in POLICIES], png)


def fig4_panels(rows):
    """The histories of Figure 4(a) (K = 5, seeds 0-4, beta grid) and 4(b) (beta = 0.25, all K and seeds)."""
    by_beta = [r for r in rows if r["K"] == 5 and r["seed"] < 5 and any(abs(r["b"] - b) < 1e-9 for b in FIG4_BETA_GRID)]
    by_K = [r for r in rows if abs(r["b"] - FIG4_BETA_K) < 1e-9]
    return by_beta, by_K


def fig4_means(rows, key):
    """x values and mean policy ratio of every policy over the histories with that x value."""
    xs = sorted({r[key] for r in rows})
    return np.array(xs), {n: np.array([np.mean([r[n] for r in rows if r[key] == v]) for v in xs]) for n in POLICIES}


def figure4(rows, out_dir, png=False):
    """policy_ratio_cv_b.pdf, policy_ratio_cv_K.pdf and legend_cv.pdf."""
    by_beta, by_K = fig4_panels(rows)
    for name, sub, key, xlabel in [("policy_ratio_cv_b", by_beta, "b", r"repair ratio $\beta$"),
                                   ("policy_ratio_cv_K", by_K, "K", r"number of trajectories $K$")]:
        if len({r[key] for r in sub}) < 2:
            continue
        fig, ax = plt.subplots(figsize=PANEL)
        fig.subplots_adjust(**ADJ)
        x, m = fig4_means(sub, key)
        for n in POLICIES:
            ax.plot(x, m[n], color=COLOR[n], marker=MARKER[n], ms=3.5, lw=1.2, label=LABEL[n])
        ax.set_xlabel(xlabel); ax.set_ylabel("Policy ratio"); ax.grid(alpha=0.25, lw=0.5)
        if key == "b":  # piecewise-linear axis: beta <= 0.3 at full scale, beta > 0.3 compressed by 2
            fwd = lambda v: np.where(np.asarray(v, float) <= 0.3, v, 0.3 + (np.asarray(v, float) - 0.3) / 2)  # noqa: E731
            inv = lambda u: np.where(np.asarray(u, float) <= 0.3, u, 0.3 + (np.asarray(u, float) - 0.3) * 2)  # noqa: E731
            ax.set_xscale("function", functions=(fwd, inv)); ax.set_xlim(-0.01, 0.71)
            ticks = [0, 0.1, 0.2, 0.3, 0.5, 0.7]
            ax.set_xticks([0.05, 0.15, 0.25, 0.4, 0.6], minor=True)
            ax.set_ylim(0.4, 1.04)
        else:
            ticks = list(x)
            ax.set_ylim(0.6, 1.04)
        ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks])
        _save(fig, out_dir, name, png)
    _legend_strip(out_dir, "legend_cv",
                  [Line2D([], [], color=COLOR[n], marker=MARKER[n], ms=3.5, lw=1.2) for n in POLICIES], png)
