"""Replication package for "Robust Weakly Coupled Markov Decision Processes"
(Fupeng Sun, Huikang Liu, Zhe Liu, Wolfram Wiesemann).

Modules
-------
instance      fleet maintenance instance of Section 5 (one machine = one constituent MDP)
data          historical trajectories, transition counts and the maximum likelihood estimate (MLE)
ambiguity     l_inf-ball ambiguity sets: worst-case kernels, lower envelope, radii of Lemma 1
lp            fluid linear program FLP(p) and CDR linear program, optimistic fluid LP
relaxation    robust Lagrangian relaxation (backward recursion) and the subgradient method (Algorithm 1)
policies      MLE, true-kernel, CDR, subgradient (robust index) and random policies
simulation    Monte Carlo evaluation under the true kernel with feasibility restoration
validation    selection of the radius multiplier by repeated cross-validation
certificates  separable worst-case value, worst-case consumption and restoration allowance
evaluation    the evaluation protocol shared by all experiments (seeds, benchmark, policy ratios)
results_io    readers for the stored result files in results/
plotting      the figures of Section 5

Submodules are imported explicitly (``from rwcmdp import lp``); importing the package itself loads nothing,
so that the plotting code runs without SciPy.
"""
__version__ = "1.0.0"
