# Stored results

All files are Python pickles of a list of dicts (one record per history or case). They are the results from which the
paper's figures, Table 2 and the numbers of Section 5 were produced; `make_figures.py` reads them. Policy ratios are
mean total rewards under the true kernel divided by that of the true-kernel policy. The keys follow the original
experiment code: `F-MLE` = MLE policy, `F-CDR` = CDR policy, `F-Sub` = subgradient policy, `random` = random policy,
`(cv)` = with the cross-validated radius multiplier, `b` = repair ratio beta, `L` = trajectory length ell.

The column "env" gives the software environment in which the file was computed (see the main README): A = Python 3.9,
numpy 2.0.2, scipy 1.13.1; B = Python 3.12, numpy 2.5.4, scipy 1.18.1. Rerunning a script in the stated environment
reproduces the stored values exactly.

| file | paper item | env | records |
|---|---|---|---|
| `fig3_lp_part1.pkl` | Fig. 3 (MLE, CDR, random) | A | 80 histories (K in {5,10,15,20}, seeds 0-19, beta = 0.5), c in {0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10} |
| `fig3_lp_part2.pkl` | Fig. 3 (MLE, CDR, random) | A | the same histories, c in {0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.25, 2.5, 4} |
| `fig3_sub.pkl` | Fig. 3 (subgradient policy) | B | the same histories, c in {0, 0.1, ..., 5} (17 values) |
| `fig4_lp.pkl` | Fig. 4 (MLE, CDR, random) | A (B for c in {4, 5, 7, 10}, see below) | 120 histories: K = 5, beta in {0.05, ..., 0.7}, seeds 0-4; beta = 0.25, K in {5, ..., 40}, seeds 0-9 |
| `fig4_lp_beta0.pkl` | Fig. 4(a), beta = 0 | A | 1 history (K = 5, seed 0); no repairs, all policies coincide |
| `fig4_sub.pkl` | Fig. 4 (subgradient policy) | B | the 120 histories of `fig4_lp.pkl` |
| `table2_certificates.pkl` | Table 2 | B | K in {5, 20}, beta in {0.5, 0.25}, c in {0.5, 2}, seeds 0-2 (the table uses c = 2) |
| `table2_sub_index.pkl` | Table 2, realized reward of the subgradient policy | B | the 12 cases with c = 2: `K, seed, b, c, v_true, realized_sub_index` (index-priority period-by-period restoration, relative to `v_true`) |
| `text_controls.pkl` | Section 5 text and footnote | B | kind `fig4` (120), `sticky` (10), `upper` (12, not used) |
| `valid_radius.pkl` | Section 5, valid radii | B | beta in {0.5, 0.25}, K in {5, 10, 15, 20}, seeds 0-19 |

## Record formats

`fig3_lp_part*.pkl`: `K, seed, b, T, L, F-MLE, random, curves`, with `curves[c] = {'F-CDR': ratio, 'F-Sub': ...}`.

`fig3_sub.pkl`: `kind = 'fig3', K, seed, b, curves`, with `curves[c]` = ratio of the subgradient policy.

`fig4_lp.pkl`: `K, seed, b, T, L, grid, cv_scores, c_cdr, F-MLE, random, F-CDR(cv), ...`. `grid` is the radius grid
{0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10}; `cv_scores[1]` the mean validation reward of the CDR policy for every grid
point, `c_cdr` the selected multiplier. (`check_diff` is internal bookkeeping of the run.)

`fig4_sub.pkl`: `kind = 'fig4', K, seed, b, grid, cv_scores, c_sub, ratio` (subgradient policy with the selected `c_sub`).

`table2_certificates.pkl`: `K, seed, b, c, v_true` (mean reward of the true-kernel policy), `coverage` (share of
state-action pairs whose ambiguity set contains the true row), `covered_all`, and for each of `MLE`, `Sub`, `CDR`:
`plan` (planning value), `vrob` (separable worst-case value), `allowance` (restoration allowance), `cert = vrob -
allowance`, `realized_fluid` (no restoration), `realized_sticky`, `frozen_sticky` (reverted machines at T),
`realized_period` (period-by-period restoration), `max_overrequest`, `Cbar` (worst-case consumption per period);
all values except `Cbar`, `frozen_sticky` and `max_overrequest` relative to `v_true`. All `realized_*` fields simulate
the policy's request probabilities with random reverts; for the subgradient policy, Table 2 instead reports
`realized_sub_index` of `table2_sub_index.pkl` (reverts by smallest index gain, Definition 5). Its `v_true` equals
that of `table2_certificates.pkl`. A rerun of `experiments/table2_certificates.py` stores this value as
`Sub.realized_index`.

`text_controls.pkl`: kind `fig4`: `K, seed, b, MLE-index, TRUE-index, random-working` (and `smoothed-MLE`, a control
that the paper does not report); kind `sticky`: `K, seed, b, c_sub, c_cdr` and `MLE | CDR | Sub: {period, sticky}`;
kind `upper`: an optimistic fluid bound that the paper does not report.

`valid_radius.pkl`: `K, seed, b, m, M, covered, rho_median, c_equiv, share_full, MLE, Sub, CDR` (see
`experiments/valid_radius.py`).

## Superseded fields

The LP files (`fig3_lp_part*.pkl`, `fig4_lp*.pkl`) also contain fields of an earlier version of the subgradient policy
(repair iff the robust index at the subgradient multiplier is nonnegative, excess requests reverted at random):
`curves[c]['F-Sub']`, `F-Sub(cv)`, `c_sub` and `cv_scores[0]`. The paper uses the subgradient policy of Definition 5,
stored in `fig3_sub.pkl` and `fig4_sub.pkl`; `rwcmdp.results_io` ignores the superseded fields (except for the beta = 0
history, where all policies coincide). The rerun scripts do not produce them.

## Provenance

The files were produced by the original experiment code from which `rwcmdp/` was assembled; their original names are
`final_fig3.pkl`, `final_fig3b.pkl`, `final_subBS.pkl` (split here by `kind` into `fig3_sub.pkl` and `fig4_sub.pkl`),
`final_fig4h.pkl`, `final_fig4f.pkl` (only its beta = 0 record is kept), `cert_check_azuma.pkl`, `controls.pkl` and
`valid_radius.pkl`. The records are unchanged. `fig4_lp.pkl` extended an earlier run with the grid {0, ..., 3} by
scoring the points {4, 5, 7, 10} with the same splits and simulation seeds; this equals a run with the full grid, which
is what `experiments/fig4_cross_validation.py` does. The original grid was scored in environment A and the extension
in environment B, so a rerun of the CDR part in environment A reproduces `cv_scores[1]` exactly on {0, ..., 3} but not
on {4, 5, 7, 10} (differences of up to 3 reward units out of about 1,000 for K = 5, seed 0, beta = 0.25); the CDR
selection lies in {4, 5, 7, 10} for three histories (K = 5, beta in {0.05, 0.1}).
