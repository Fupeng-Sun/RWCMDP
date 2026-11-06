"""Regenerate Figures 3 and 4, Table 2 and the numbers quoted in Section 5 from the stored results (no reruns).

Usage:  python make_figures.py [--results results] [--figures figures] [--png]
Writes figures/policy_ratio_K{5,10,15,20}.pdf, legend_radius.pdf, policy_ratio_cv_b.pdf, policy_ratio_cv_K.pdf,
legend_cv.pdf, table2_certificates.tex and numbers.txt (the numbers of the text, also printed). Sections whose result
files are missing in --results are skipped, so the script also summarizes partial reruns (e.g. --results output).
Figure 1 (Example 1) is drawn by pgfplots in the paper; its coordinates come from experiments/fig1_nonconvex.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwcmdp import plotting  # noqa: E402
from rwcmdp.results_io import load, load_fig3, load_fig4  # noqa: E402

LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def pt(x):
    """Difference of two policy ratios in percentage points."""
    return 100 * x


def fig3_numbers(fig3):
    say("Figure 3 (beta = 0.5; medians over the histories)")
    for K in sorted({k for k, _ in fig3}):
        d = plotting.fig3_data(fig3, K)
        c = d["c"]
        med = {n: np.median(d[n], axis=0) for n in ("sub", "cdr")}
        q1, q3 = {}, {}
        for n in ("sub", "cdr"):
            q1[n], q3[n] = np.percentile(d[n], [25, 75], axis=0)
        j5 = int(np.argmin(np.abs(c - 0.5)))
        say(f"  K={K:2d} ({len(d['mle'])} histories, {len(c)} radii): MLE {np.median(d['mle']):.3f}, random "
            f"{np.median(d['random']):.3f}; at c=0.5 Sub {med['sub'][j5]:.3f} CDR {med['cdr'][j5]:.3f}; "
            f"peak Sub {med['sub'].max():.3f} (c={c[med['sub'].argmax()]:g}) CDR {med['cdr'].max():.3f} "
            f"(c={c[med['cdr'].argmax()]:g}); at c={c[-1]:g} Sub {med['sub'][-1]:.3f} CDR {med['cdr'][-1]:.3f}")
        iqr_mle = np.subtract(*np.percentile(d["mle"], [75, 25]))
        big = c >= 0.5
        say(f"         IQR (pt): MLE {pt(iqr_mle):.2f}; c=0 Sub {pt(q3['sub'][0] - q1['sub'][0]):.2f} CDR "
            f"{pt(q3['cdr'][0] - q1['cdr'][0]):.2f}; max over c>=0.5 Sub {pt((q3['sub'] - q1['sub'])[big].max()):.2f} "
            f"CDR {pt((q3['cdr'] - q1['cdr'])[big].max()):.2f}; gain of the best robust median over MLE "
            f"{pt(max(med['sub'].max(), med['cdr'].max()) - np.median(d['mle'])):.1f} pt; at c={c[-1]:g} over MLE: "
            f"Sub {pt(med['sub'][-1] - np.median(d['mle'])):.1f} CDR {pt(med['cdr'][-1] - np.median(d['mle'])):.1f} pt")


def cv_regret(fig3, fig4):
    """Loss of the validated multiplier relative to the best multiplier in hindsight (K = 5, beta = 0.5)."""
    rows = [r for r in fig4 if r["K"] == 5 and abs(r["b"] - 0.5) < 1e-9 and (5, r["seed"]) in fig3]
    if not rows:
        return
    reg = {"sub": [], "cdr": []}
    for r in rows:
        d = fig3[(5, r["seed"])]
        for n in reg:
            reg[n].append(max(d[n].values()) - r[n])
    say(f"Cross-validation regret vs best c in hindsight (K=5, beta=0.5, {len(rows)} histories, Figure 3 radii): "
        f"Sub mean {pt(np.mean(reg['sub'])):.2f} max {pt(np.max(reg['sub'])):.2f} pt; "
        f"CDR mean {pt(np.mean(reg['cdr'])):.2f} max {pt(np.max(reg['cdr'])):.2f} pt")


def fig4_numbers(fig4, controls):
    by_beta, by_K = plotting.fig4_panels(fig4)
    if by_beta:
        x, m = plotting.fig4_means(by_beta, "b")
        say("Figure 4(a) (K = 5, means over the histories)")
        for j, b in enumerate(x):
            n = sum(abs(r["b"] - b) < 1e-9 for r in by_beta)
            say(f"  beta={b:<4g} ({n}): MLE {m['mle'][j]:.3f} Sub {m['sub'][j]:.3f} CDR {m['cdr'][j]:.3f} "
                f"random {m['random'][j]:.3f}")
        pos = x > 0
        say(f"  Sub - 1: {pt((m['sub'] - 1)[pos].min()):.1f} to {pt((m['sub'] - 1)[pos].max()):.1f} pt; Sub - MLE: "
            f"{pt((m['sub'] - m['mle'])[pos].min()):.1f} to {pt((m['sub'] - m['mle'])[pos].max()):.1f} pt; "
            f"CDR min for beta >= 0.3: {m['cdr'][x >= 0.3 - 1e-9].min():.3f}; CDR - MLE for 0 < beta <= 0.15: "
            + ", ".join(f"{pt(v):.1f}" for v in (m['cdr'] - m['mle'])[pos & (x <= 0.15 + 1e-9)]) + " pt")
    if by_K:
        x, m = plotting.fig4_means(by_K, "K")
        say("Figure 4(b) (beta = 0.25, means over the histories)")
        for j, K in enumerate(x):
            n = sum(r["K"] == K for r in by_K)
            say(f"  K={K:2d} ({n}): MLE {m['mle'][j]:.3f} Sub {m['sub'][j]:.3f} CDR {m['cdr'][j]:.3f} "
                f"random {m['random'][j]:.3f}   CDR-MLE {pt(m['cdr'][j] - m['mle'][j]):5.1f} pt, "
                f"Sub-MLE {pt(m['sub'][j] - m['mle'][j]):5.1f} pt")
    ctrl = [r for r in controls if r["kind"] == "fig4" and abs(r["b"] - 0.25) < 1e-9]
    if ctrl and by_K:
        say("Controls (beta = 0.25; mean / median over the histories)")
        sub = {(r["K"], r["seed"]): r["sub"] for r in by_K}
        for K in sorted({r["K"] for r in ctrl}):
            rr = [r for r in ctrl if r["K"] == K]
            line = f"  K={K:2d}: " + ", ".join(
                f"{k} {np.mean([r[k] for r in rr]):.4f} / {np.median([r[k] for r in rr]):.4f}"
                for k in ("MLE-index", "TRUE-index", "random-working"))
            if all((K, r["seed"]) in sub for r in rr):
                line += f"; Sub - MLE-index {pt(np.mean([sub[(K, r['seed'])] - r['MLE-index'] for r in rr])):.2f} pt"
            say(line)


def sticky_numbers(controls):
    rows = [r for r in controls if r["kind"] == "sticky"]
    if not rows:
        return
    say("Sticky vs period-by-period restoration (K = 5, validated c): largest drop over the histories (pt)")
    for b in sorted({r["b"] for r in rows}):
        rr = [r for r in rows if r["b"] == b]
        say(f"  beta={b:g} ({len(rr)}): " + ", ".join(
            f"{n} {pt(max(r[n]['period'] - r[n]['sticky'] for r in rr)):.2f}" for n in ("MLE", "CDR", "Sub"))
            + "; Sub sticky min " + f"{min(r['Sub']['sticky'] for r in rr):.3f}")


def valid_radius_numbers(rows):
    if not rows:
        return
    say("Valid radii of Section 4.2 (alpha = 0.05; coverage and full-simplex share = means, ratios = medians)")
    for b in sorted({r["b"] for r in rows}, reverse=True):
        for K in sorted({r["K"] for r in rows}):
            rr = [r for r in rows if r["K"] == K and r["b"] == b]
            med = lambda k: np.median([r[k] for r in rr])  # noqa: E731
            say(f"  beta={b:<4g} K={K:2d} ({len(rr)}): M={rr[0]['M']} m={rr[0]['m']} coverage "
                f"{np.mean([r['covered'] for r in rr]):.3f} full simplex {100 * np.mean([r['share_full'] for r in rr]):.0f}% "
                f"median radius {med('rho_median'):.2f} (c = rho sqrt(m) {med('c_equiv'):.1f}) | MLE {med('MLE'):.3f} "
                f"Sub {med('Sub'):.3f} CDR {med('CDR'):.3f}")


def table2(rows, sub_index, out_dir):
    """Table 2; the realized reward of the subgradient policy uses index-priority restoration ('realized_index' of a
    rerun, or table2_sub_index.pkl for the stored results, whose 'realized_period' used random reverts)."""
    rows = [r for r in rows if abs(r["c"] - 2.0) < 1e-9]
    if not rows:
        return
    idx = {(r["K"], r["seed"], r["b"], r["c"]): r["realized_sub_index"] for r in sub_index}
    for r in rows:
        if "realized_index" not in r["Sub"]:
            key = (r["K"], r["seed"], r["b"], r["c"])
            if key not in idx:
                say("Table 2: index-priority values of the subgradient policy missing; showing random reverts")
                idx = None
                break
            r["Sub"] = dict(r["Sub"], realized_index=idx[key])
    for r in rows:
        r["Sub"]["realized_period_table"] = r["Sub"]["realized_index"] if idx is not None else r["Sub"]["realized_period"]
        for p in ("MLE", "CDR"):
            r[p]["realized_period_table"] = r[p]["realized_period"]
    say("Table 2 (c = 2; means over the histories): plan / worst case / realized (period-by-period restoration;"
        " subgradient policy by index priority)")
    tex = []
    for K in sorted({r["K"] for r in rows}):
        for b in sorted({r["b"] for r in rows}, reverse=True):
            rr = [r for r in rows if r["K"] == K and r["b"] == b]
            vals = [np.mean([r[p][k] for r in rr]) for p in ("MLE", "Sub", "CDR") for k in ("plan", "vrob", "realized_period_table")]
            say(f"  K={K:2d} beta={b:<4g} ({len(rr)}): " + " | ".join(
                " ".join(f"{v:.3f}" for v in vals[3 * j:3 * j + 3]) for j in range(3)))
            tex.append(f"${K}$ & ${b:g}$ & " + " & ".join(f"${v:.3f}$" for v in vals) + r" \\")
    with open(os.path.join(out_dir, "table2_certificates.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    m = {(r["K"], r["b"]): r for r in rows}
    gaps = {K: [np.mean([r["MLE"]["plan"] - r["MLE"]["realized_period_table"] for r in rows if r["K"] == K and r["b"] == b])
                for b in (0.5, 0.25) if (K, b) in m] for K in sorted({k for k, _ in m})}
    say("  MLE plan - realized (pt): " + "; ".join(f"K={K}: " + ", ".join(f"{pt(v):.1f}" for v in g) for K, g in gaps.items()))
    cases = [(K, b) for K in sorted({r["K"] for r in rows}) for b in sorted({r["b"] for r in rows}, reverse=True)]
    mean = lambda K, b, p, k: np.mean([r[p][k] for r in rows if r["K"] == K and r["b"] == b])  # noqa: E731
    below = [(K, b, p) for K, b in cases for p in ("MLE", "Sub", "CDR") if mean(K, b, p, "vrob") > mean(K, b, p, "realized_period_table")]
    say(f"  worst case > realized in {len(below)} of {3 * len(cases)} cases: {below}; CDR plan <= realized in all cases: "
        f"{all(mean(K, b, 'CDR', 'plan') <= mean(K, b, 'CDR', 'realized_period_table') for K, b in cases)}; "
        f"max |Sub plan - worst case| {max(abs(mean(K, b, 'Sub', 'plan') - mean(K, b, 'Sub', 'vrob')) for K, b in cases):.1e}")
    k5 = [(b, p) for b in (0.5, 0.25) for p in ("Sub", "CDR") if (5, b) in m]
    say("  K=5: robust worst case - MLE worst case (pt): "
        + ", ".join(f"beta={b:g} {p} {pt(mean(5, b, p, 'vrob') - mean(5, b, 'MLE', 'vrob')):.1f}" for b, p in k5))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(here, "results"))
    p.add_argument("--figures", default=os.path.join(here, "figures"))
    p.add_argument("--png", action="store_true", help="also write PNG versions of the figures")
    a = p.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    fig3, fig4 = load_fig3(a.results), load_fig4(a.results)
    controls = load(a.results, "text_controls")
    if fig3:
        plotting.figure3(fig3, a.figures, a.png)
        fig3_numbers(fig3)
    if fig4:
        plotting.figure4(fig4, a.figures, a.png)
        cv_regret(fig3, fig4)
        fig4_numbers(fig4, controls)
    sticky_numbers(controls)
    valid_radius_numbers(load(a.results, "valid_radius"))
    table2(load(a.results, "table2_certificates"), load(a.results, "table2_sub_index"), a.figures)
    with open(os.path.join(a.figures, "numbers.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nfigures, table2_certificates.tex and numbers.txt written to {a.figures}")
