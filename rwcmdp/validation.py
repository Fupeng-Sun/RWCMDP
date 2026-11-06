"""Selection of the radius multiplier c by repeated cross-validation (Section 5, second experiment).

In each of three folds, a random 60% of the observed transitions of every machine gives the training MLE, from which
the policy is constructed, and the remaining 40% give the validation kernel (the MLE of the held-out transitions),
under which the policy is simulated (300 runs). The multiplier with the largest mean validation reward over the folds
is selected (ties: the smallest multiplier) and the policy is then constructed from all observed transitions.
"""
import numpy as np

from . import lp
from .ambiguity import radius
from .data import mle_kernel, transition_counts
from .evaluation import CV_RUNS, capacity
from .instance import HORIZON
from .policies import cdr_policy, subgradient_policy
from .simulation import simulate, simulate_index

N_FOLDS = 3
TRAIN_FRACTION = 0.6
RADIUS_GRID = (0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)


def holdout_split(h, fold, train_fraction=TRAIN_FRACTION):
    """Training and validation MLEs of a fold: a random share of every machine's transitions (stream [3, seed, K, fold])."""
    I = h.n_machines
    n_trans = h.states[:, :, :-1].size // I
    n_train = int(round(train_fraction * n_trans))
    keys = np.random.default_rng([3, h.seed, h.K, fold]).random((I, n_trans))
    train = (np.argsort(np.argsort(keys, axis=1), axis=1) < n_train).reshape(h.states[:, :, :-1].shape)
    S = h.P0.shape[1]
    P_train = mle_kernel(transition_counts(h.states, h.actions, S, train), h.mask)
    P_valid = mle_kernel(transition_counts(h.states, h.actions, S, ~train), h.mask)
    return P_train, P_valid


def validation_scores(h, beta, policy, grid=RADIUS_GRID, n_folds=N_FOLDS, horizon=HORIZON):
    """Mean validation reward over the folds for every multiplier in ``grid``; policy is 'sub' or 'cdr'."""
    cap = capacity(beta, h)
    scores = np.zeros((n_folds, len(grid)))
    for fold in range(n_folds):
        P_tr, P_va = holdout_split(h, fold)
        if policy == "sub":   # starting point of Algorithm 1: the budget multipliers of FLP(training MLE)
            _, _, mu_tr = lp.solve_fluid_lp(P_tr, h.R, h.q, beta, horizon)
        stream = [4, h.seed, h.K, fold]
        for k, c in enumerate(grid):
            rho = radius(c, h.K, h.L)
            if policy == "sub":
                pol = subgradient_policy(P_tr, rho, h.R, h.q, beta, h.mask, horizon, mu_tr)
                scores[fold, k] = simulate_index(stream, pol, P_va, h.R, h.q, cap, CV_RUNS)
            elif policy == "cdr":
                _, prob = cdr_policy(P_tr, rho, h.R, h.q, beta, h.mask, horizon)
                scores[fold, k] = simulate(stream, prob, P_va, h.R, h.q, cap, CV_RUNS)
            else:
                raise ValueError(policy)
    return scores.mean(axis=0)


def select_multiplier(scores, grid=RADIUS_GRID):
    """The multiplier with the largest mean validation reward (the first one in case of ties)."""
    return float(np.asarray(grid)[int(np.argmax(scores))])
