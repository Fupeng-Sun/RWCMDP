# Robust Weakly Coupled Markov Decision Processes: replication package

Code, data and results for

> Fupeng Sun, Huikang Liu, Zhe Liu, Wolfram Wiesemann. *Robust Weakly Coupled Markov Decision Processes.*

The package reproduces every numerical result of the paper: Figure 1 (Example 1, nonconvexity of the robust
Lagrangian bound), Figures 3 and 4 and Table 2 (fleet maintenance experiments of Section 5), the numbers quoted in the
text of Section 5, and the simulation mentioned at the end of Section 4.4. Figure 2 (the single-machine MDP) is a
TikZ drawing of the instance defined in `rwcmdp/instance.py`.

## Contents

```
README.md               this file
LICENSE                 MIT license
requirements.txt        environment A (Python 3.9): numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4
requirements-b.txt      environment B (Python 3.12): numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2
make_figures.py         Figures 3-4, Table 2 and the text numbers from results/ (no reruns, a few seconds)
rwcmdp/                 the methods (importable package)
experiments/            one script per paper item; each writes its records to output/ (or --out-dir)
results/                the stored results used in the paper (see results/README.md)
figures/                the figures, Table 2 and numbers.txt regenerated from results/
```

`rwcmdp/` modules and their counterparts in the paper:

| module | content |
|---|---|
| `instance.py` | fleet maintenance instance (Section 5, Figure 2): kernel, rewards, supports, budgets beta * I |
| `data.py` | historical trajectories under the behavior policy, transition counts, MLE (Section 4.2) |
| `ambiguity.py` | l_inf-ball ambiguity sets on the simplex of reachable states: worst-case kernels (inner minimization), lower envelope, radii rho_{m,i}(s,a) of Lemma 1 |
| `lp.py` | fluid LP FLP(p) for kernels and kernel sequences, CDR LP with the scaled budget (HiGHS via SciPy) |
| `relaxation.py` | robust Lagrangian relaxation (backward recursion, subgradients) and the subgradient method (Algorithm 1) |
| `policies.py` | fluid policies (Definition 1), MLE and true-kernel policies (Definition 4), CDR policy, subgradient policy = optimal Lagrangian index policy for the kernel sequence p~ (Definition 5), random policies |
| `simulation.py` | Monte Carlo evaluation under the true kernel with period-by-period or sticky feasibility restoration (Definition 3) |
| `validation.py` | selection of the radius multiplier by repeated cross-validation (three random 60/40 splits) |
| `certificates.py` | separable worst-case value, worst-case consumption and restoration allowance (Proposition 3, eq. (EC.allowance)) |
| `evaluation.py` | the evaluation protocol: random streams, 1,000 simulation runs, policy ratios |
| `results_io.py`, `plotting.py` | readers of the stored results and the figures |

## Requirements and software environments

Python 3.9 or later with NumPy, SciPy and Matplotlib; no other packages. The fluid LPs are solved with the HiGHS
solver shipped with SciPy. Install with `pip install -r requirements.txt` (or `requirements-b.txt`).

The fluid LPs of the fleet have many optimal solutions (the machines are identical), and different HiGHS versions can
return different optimal solutions and multipliers. The policies built from them, and hence the simulated policy
ratios, then differ slightly (we observed differences up to 0.1 percentage points for a single history). The stored
results were computed in two environments, and each result file is reproduced exactly (to the last digit) in its
environment:

* environment A, `requirements.txt` (Python 3.9.6, numpy 2.0.2, scipy 1.13.1): the MLE, CDR and random policies of
  Figures 3 and 4 (`fig3_lp*.pkl`, `fig4_lp*.pkl`; in `fig4_lp.pkl`, the validation scores of the CDR policy for
  c in {4, 5, 7, 10} and the three CDR ratios whose validated c lies in this set were computed in environment B);
* environment B, `requirements-b.txt` (Python 3.12.13, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2): the
  subgradient policy of Figures 3 and 4 (`fig3_sub.pkl`, `fig4_sub.pkl`), Table 2 (both files), the controls, the valid-radius
  experiment, and the figure PDFs (`make_figures.py` in environment B reproduces the paper's PDFs pixel by pixel;
  matplotlib 3.9.4 draws the same data with slightly different glyph rendering).

All code runs in both environments.

## Data

All data are simulated; there are no external data. Every random quantity is drawn from
`numpy.random.default_rng(list)` with a fixed key (see `rwcmdp/evaluation.py`):

| key | use |
|---|---|
| `[7, seed]` | the historical trajectories of history `seed` (K trajectories of ell = 15 periods per machine, behavior policy repairs with probability 0.3) |
| `[1, seed]` | out-of-sample evaluation: 1,000 runs of the fleet under the true kernel, common to all policies, radii and K |
| `[3, seed, K, fold]` | random 60/40 split of the observed transitions in a validation fold |
| `[4, seed, K, fold]` | 300 runs under the validation kernel of a fold |

The instance: I = 100 machines, T = 30 periods, nine states per machine, shock failure probability 0.04 (s - 1) at age
s, uniform initial distribution, repair capacity beta * I. The MLE uses uniform rows on the known support for
unobserved pairs; the ambiguity set of every pair is the l_inf ball of radius rho = c / sqrt(K ell) around the MLE
within the simplex over the reachable states; Algorithm 1 runs 150 iterations with step sizes 0.01 / sqrt(j) from the
budget multipliers of FLP(p_hat).

## Reproducing the paper

Regenerate all figures, Table 2 and the numbers of Section 5 from the stored results (a few seconds):

```
python make_figures.py            # writes figures/ and prints the numbers (also in figures/numbers.txt)
```

To recompute results, run the scripts in `experiments/`. Each takes `--workers` (default 6) and `--out-dir` (default
`output/`) and options to run a subset (`--K`, `--beta`, `--seeds`, `--radii`, ...; see `--help`). A rerun is
summarized with `python make_figures.py --results output --figures output/figures`.

| paper item | script | result file(s) | full run (6 workers, Apple M5 Pro) |
|---|---|---|---|
| Fig. 1(b), Table EC.1 | `experiments/fig1_nonconvex.py` | printed coordinates (exact arithmetic) | < 1 s |
| Fig. 3 | `experiments/fig3_radius.py` | `fig3_lp_part1.pkl`, `fig3_lp_part2.pkl`, `fig3_sub.pkl` | about 2-3 h |
| Fig. 4 | `experiments/fig4_cross_validation.py` | `fig4_lp.pkl`, `fig4_lp_beta0.pkl`, `fig4_sub.pkl` | about 1 day (30-45 min per history and policy) |
| Table 2 | `experiments/table2_certificates.py` (stored Sub column: `--sub-index-only`) | `table2_certificates.pkl`, `table2_sub_index.pkl` | a few minutes |
| Section 5: index policies from p_hat and p0, random policy for working machines, footnote on sticky restoration | `experiments/text_controls.py` | `text_controls.pkl` | about 10 min |
| Section 5: valid radii (alpha = 0.05) | `experiments/valid_radius.py` | `valid_radius.pkl` | about 10 min |
| Section 5: cross-validation regret | `make_figures.py` (from Fig. 3 and 4 results) | - | - |
| Section 5: binding periods (23 and 7 of 30), unobserved pairs (12% for K = 5), shock-free MLE rows | `experiments/instance_stats.py` | printed | 10 s |
| Section 5: runtimes (fluid LP, CDR LP, Algorithm 1, simulation) | `experiments/timing.py` | printed | 1 min |
| Section 4.4: revert order with asymmetric plans | `experiments/sec44_tie_order.py` | printed | 5 s |

Quick checks (one history each; about 1 min each, except Figure 4: 35-45 min per policy):

```
python experiments/fig3_radius.py --K 5 --seeds 0 --radii 0 0.5 2 --workers 2
python experiments/fig4_cross_validation.py --K 5 --beta 0.25 --seeds 0 --workers 2
python experiments/table2_certificates.py --K 5 --beta 0.5 --c 2 --seeds 0 --workers 1
python experiments/text_controls.py --K 5 --beta 0.25 --seeds 0 --workers 2
python experiments/valid_radius.py --K 5 --beta 0.5 --seeds 0 --workers 1
```

Runtimes depend on the LP solver; most of the time goes into the CDR LPs (about 13 s each, longer for large radii)
and, for Figure 4, into the 33 policies (3 folds x 11 multipliers) per history and policy that the validation
constructs. The original runs used 12-18 worker processes.

## Conventions

* Policy ratio = mean total reward of a policy under the true kernel / mean total reward of the true-kernel policy,
  both over the same 1,000 simulation runs (stream `[1, seed]`).
* Restoration in the experiments is period by period: LP-based policies (MLE, CDR, true kernel, random) revert excess
  repair requests uniformly at random, the subgradient policy reverts the smallest index gains; `simulation.py` also
  implements the sticky restoration of the analysis (used in the footnote of Section 5 and in Table 2's
  `realized_sticky`).
* In Figures 3 and 4 the CDR LP states the flow constraints as inequalities, in Table 2 as equalities (the definition of
  the CDR policy); since rewards are nonnegative, both have the same optimal value. In the histories we checked, the
  solutions returned with inequality flows have zero slack (below 3e-16), so they are optimal solutions with equality
  flows as the definition requires; solving with equality flows can return another optimal solution (the LP is
  degenerate), whose policy ratio differed by at most 0.07 percentage points.
* Table 2 reports realized rewards with period-by-period restoration: random reverts for the MLE and CDR policies and
  index-priority reverts for the subgradient policy (`results/table2_sub_index.pkl`, field `realized_sub_index`;
  `experiments/table2_certificates.py` writes it as `Sub.realized_index`). The stored `table2_certificates.pkl`
  predates this field; its `Sub.realized_period` simulates random reverts and is 0.04-0.39 percentage points lower.

## License

MIT, see `LICENSE`.
