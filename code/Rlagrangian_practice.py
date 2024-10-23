from __future__ import annotations

import concurrent.futures
import math
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import cplex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cplex import Cplex, SparsePair
from cplex.exceptions import CplexError
from tqdm.auto import tqdm


def generate_dist(dimension, lower_bound):
    """
    Generate a probability distribution for a given dimension.

    Args:
    dimension (int): The dimension of the distribution.
    lower_bound (float): The lower bound for each element before normalization.

    Returns:
    None
    """
    dist = list(np.random.rand(1, dimension) + lower_bound)
    dist = dist / np.sum(dist)
    return list(dist[0])


def generate_random_transition_kernel(dimension):
    # Step 1: Create an n x n matrix of random numbers
    matrix = np.random.rand(dimension, dimension)

    # Step 2: Normalize each row to make the row sum to 1
    row_sums = matrix.sum(axis=1)
    transition_kernel = matrix / row_sums[:, np.newaxis]

    return transition_kernel


def generate_number_in_dist_batch(dists):
    """
    Generate a batch of numbers based on multiple probability distributions (one per bandit) using a vectorized approach.

    Args:
        dists (np.ndarray): A 2D array where each row is a probability distribution for a bandit (shape: [number_of_bandits, number_of_states]).

    Returns:
        np.ndarray: An array of indices representing the new states for each bandit.
    """
    new_states = np.array(
        [
            np.random.choice(dists.shape[1], p=dists[i])
            for i in range(dists.shape[0])
        ]
    )

    return new_states


def sort_truncated(
    weight, number_of_bandit_in_total, maximum_number_of_pulled_bandits
):
    """
    Sorts the weights and generates a feasible action based on the sorted weights.

    Args:
    weight (list): A list of weights.
    action (list): A list representing the action vector.
    number_of_bandit_in_total (int): The total number of bandits.
    maximum_number_of_pulled_bandits (int): The maximum number of bandits that can be pulled.

    Returns:
    The sum of pulled bandits (int), and feasible action vector (list).
    """
    indices = sorted(
        range(number_of_bandit_in_total), key=lambda i: weight[i], reverse=True
    )[:maximum_number_of_pulled_bandits]
    action = [
        1 if (i in indices and weight[i] > 0) else 0
        for i in range(number_of_bandit_in_total)
    ]

    return action, np.sum(action)


def random_truncated(
    weight, number_of_bandit_in_total, maximum_number_of_pulled_bandits
):
    """
    Generate a random truncated feasible action based on the weights.

    Args:
    weight (list): A list of weights.
    action (list): A list representing the action vector.
    number_of_bandit_in_total (int): The total number of bandits.
    maximum_number_of_pulled_bandits (int): The maximum number of bandits that can be pulled.

    Returns:
    The sum of pulled bandits (int), and lfeasible action vector (list).
    """
    indices = [i for i, x in enumerate(weight) if x > 0]
    if len(indices) > maximum_number_of_pulled_bandits:
        indices = random.sample(indices, maximum_number_of_pulled_bandits)
    action = [
        1 if i in indices else 0 for i in range(number_of_bandit_in_total)
    ]

    return action, np.sum(action)


def initialization(
    number_of_bandit_in_total, number_of_states, number_of_timeperiods, r
):
    # def initialization(number_of_bandit_in_total, number_of_states, number_of_timeperiods, init_prob, prob_pull, prob_donothing, reward):
    """
    Initialize the parameters for the bandit problem.

    Args:
    number_of_bandit_in_total (int): The total number of bandits.
    number_of_states (int): The number of states.
    number_of_timeperiods (int): The number of time periods.
    init_prob (list): The list to store the initial probabilities for each group.
    prob_pull (list): The list to store the transition probabilities of pull action.
    prob_donothing (list): The list to store the transition probabilities of donothing action.
    reward (list): The list to store the reward function.
    radius_pull (list): The list to store the radius of the ambuguity set of pull action.
    radius_donothing (list): The list to store the radius of the ambuguity set of donothing action.

    Returns:
    None
    """

    # Randomly generate the initial probability
    init_prob = np.array(
        [
            generate_dist(number_of_states, 0)
            for j in range(number_of_bandit_in_total)
        ]
    )

    # Transition probability
    # pull
    # prob_pull = [[[generate_dist(number_of_states, 0) for k in range(number_of_states)] for j in range(number_of_bandit_in_total)] for t in range(number_of_timeperiods)]
    prob_pull = np.array(
        [
            generate_random_transition_kernel(number_of_states)
            for _ in range(number_of_bandit_in_total)
        ]
    )

    # do nothing
    prob_donothing = [
        [
            [
                1 if item_idx == row_idx else 0
                for item_idx in range(0, number_of_states)
            ]
            for row_idx in range(0, number_of_states)
        ]
        for j in range(number_of_bandit_in_total)
    ]

    # # Randomly generate the radius in (0, 0.1)
    # radius_pull = np.random.rand(number_of_bandit_in_total, number_of_states) * 10 *r
    # radius_donothing = np.random.rand(number_of_bandit_in_total, number_of_states) * 10 *r

    radius_pull = r * np.ones(
        (number_of_timeperiods, number_of_bandit_in_total, number_of_states)
    )
    radius_donothing = r * np.ones(
        (number_of_timeperiods, number_of_bandit_in_total, number_of_states)
    )

    # Randomly generate the reward function
    reward = np.random.rand(number_of_bandit_in_total, number_of_states)
    return (
        init_prob,
        prob_pull,
        prob_donothing,
        reward,
        radius_pull,
        radius_donothing,
    )


def fluid_LP(
    T0,
    number_of_timeperiods,
    number_of_states,
    reward,
    init_prob,
    prob_pull,
    prob_donothing,
    pulled_ratio,
    number_of_bandit_in_total,
    radius_pull,
    radius_donothing,
    change,
):
    """
    Solve the fluid_LP using CPLEX.

    Args:
    # pi1_LP (list): A 3D list to store the pi variables with pull arms.
    # pi0_LP (list): A 3D list to store the pi variables with donothing action.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 2D list of rewards of pull action.
    init_prob (list): A 2D list of initial probabilities.
    prob_pull (list): A 3D list of transition probabilities of pull action.
    prob_dothing (list): A 3D list of transition probabilities of dothing action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.

    Returns:
    The objective value of the solved LP (float), and the optimal solution pi1_LP, pi0_LP.
    """
    # Initialize CPLEX environment and model
    cplex = Cplex()

    # ! we disable the stdout of CPLEX
    cplex.set_log_stream(None)
    cplex.set_results_stream(None)
    # ! and we shall limit the thread using as we use multiprocessing
    cplex.parameters.threads.set(1)

    cplex.set_problem_type(Cplex.problem_type.LP)
    cplex.objective.set_sense(cplex.objective.sense.maximize)

    # Variable and constraint initialization

    pi1 = [
        [
            ["pi1_{}_{}_{}".format(t, j, k) for k in range(number_of_states)]
            for j in range(number_of_bandit_in_total)
        ]
        for t in range(T0, number_of_timeperiods)
    ]

    pi0 = [
        [
            ["pi0_{}_{}_{}".format(t, j, k) for k in range(number_of_states)]
            for j in range(number_of_bandit_in_total)
        ]
        for t in range(T0, number_of_timeperiods)
    ]

    # Combine pi1 and pi0 variables and add them to the CPLEX model with lower bounds of 0 in one go
    all_vars = [
        var
        for time_vars in pi1
        for bandit_vars in time_vars
        for var in bandit_vars
    ] + [
        var
        for time_vars in pi0
        for bandit_vars in time_vars
        for var in bandit_vars
    ]
    cplex.variables.add(names=all_vars, lb=[0] * len(all_vars))

    # Objective function (maximize reward)
    Robjective = [
        (f"pi1_{t}_{j}_{k}", reward[j][k])
        for k in range(number_of_states)
        for j in range(number_of_bandit_in_total)
        for t in range(T0, number_of_timeperiods)
    ]
    cplex.objective.set_linear(Robjective)

    # Initial constraints
    initial_constraints = [
        SparsePair(
            ind=[
                "pi1_{}_{}_{}".format(T0, j, k),
                "pi0_{}_{}_{}".format(T0, j, k),
            ],
            val=[1, 1],
        )
        for j in range(number_of_bandit_in_total)
        for k in range(number_of_states)
    ]
    cplex.linear_constraints.add(
        lin_expr=initial_constraints,
        senses=["E"] * len(initial_constraints),
        rhs=init_prob.flatten().tolist(),
    )

    # Time period constraints
    time_constraints = []
    time_senses = []
    time_rhs = []

    for t in range(T0, number_of_timeperiods):
        curr_LHS = [
            "pi1_{}_{}_{}".format(t, j, k)
            for j in range(number_of_bandit_in_total)
            for k in range(number_of_states)
        ]
        time_constraints.append(
            SparsePair(ind=curr_LHS, val=[1] * len(curr_LHS))
        )
        time_senses.append("L")
        time_rhs.append(pulled_ratio * number_of_bandit_in_total)

        if t < number_of_timeperiods - 1:
            for j in range(number_of_bandit_in_total):
                for k in range(number_of_states):
                    curr_RHS = (
                        [
                            "pi1_{}_{}_{}".format(t + 1, j, k),
                            "pi0_{}_{}_{}".format(t + 1, j, k),
                        ]
                        + [
                            "pi1_{}_{}_{}".format(t, j, l)
                            for l in range(number_of_states)
                        ]
                        + [
                            "pi0_{}_{}_{}".format(t, j, l)
                            for l in range(number_of_states)
                        ]
                    )
                    curr_RHS_coeff = (
                        [-1, -1]
                        + [prob_pull[j][l][k] for l in range(number_of_states)]
                        + [
                            prob_donothing[j][l][k]
                            for l in range(number_of_states)
                        ]
                    )
                    time_constraints.append(
                        SparsePair(ind=curr_RHS, val=curr_RHS_coeff)
                    )
                    time_senses.append("E")
                    time_rhs.append(0)

    cplex.linear_constraints.add(
        lin_expr=time_constraints, senses=time_senses, rhs=time_rhs
    )

    # Solve the problem
    cplex.parameters.timelimit.set(60)  # Set a time limit if needed
    try:
        cplex.solve()
    except CplexError as exc:
        print(exc)
        return None

    # Retrieve the objective value
    objval = cplex.solution.get_objective_value()

    pi = (
        np.array(cplex.solution.get_values())
        .view()
        .reshape(
            2,
            number_of_timeperiods - T0,
            number_of_bandit_in_total,
            number_of_states,
        )
    )
    pi1_LP = pi[0, :, :, :]
    pi0_LP = pi[1, :, :, :]
    # # Write the objective value to the output file
    # fout.write(f"{objval}\n")

    return objval, pi1_LP, pi0_LP


def constant_decision_rule_LP(
    T0,
    number_of_timeperiods,
    number_of_states,
    reward,
    init_prob,
    prob_pull,
    prob_donothing,
    pulled_ratio,
    number_of_bandit_in_total,
    radius_pull,
    radius_donothing,
    change,
):
    """
    Solve the constant_decision_rule_LP using CPLEX.

    Args:
    # pi1_LP (list): A 3D list to store the pi variables with pull arms.
    # pi0_LP (list): A 3D list to store the pi variables with donothing action.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 3D list of rewards of pull action.
    init_prob (list): A 2D list of initial probabilities.
    prob_pull (list): A 4D list of transition probabilities of pull action.
    prob_dothing (list): A 4D list of transition probabilities of dothing action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.
    radius_pull (list): The list to store the radius of the ambuguity set of pull action.
    radius_donothing (list): The list to store the radius of the ambuguity set of donothing action.

    Returns:
    The objective value of the solved LP (float), and the optimal solution pi1_LP, pi0_LP.
    """
    # Initialize CPLEX environment and model
    cplex = Cplex()

    # ! we disable the stdout of CPLEX
    cplex.set_log_stream(None)
    cplex.set_results_stream(None)
    # ! and we shall limit the thread using as we use multiprocessing
    cplex.parameters.threads.set(1)

    cplex.set_problem_type(Cplex.problem_type.LP)
    cplex.objective.set_sense(cplex.objective.sense.maximize)

    # Variable and constraint initialization

    Rpi1 = [
        [
            ["Rpi1_{}_{}_{}".format(t, j, k) for k in range(number_of_states)]
            for j in range(number_of_bandit_in_total)
        ]
        for t in range(T0, number_of_timeperiods)
    ]

    Rpi0 = [
        [
            ["Rpi0_{}_{}_{}".format(t, j, k) for k in range(number_of_states)]
            for j in range(number_of_bandit_in_total)
        ]
        for t in range(T0, number_of_timeperiods)
    ]

    # Combine pi1 and pi0 variables and add them to the CPLEX model with lower bounds of 0 in one go
    Rall_vars = [
        var
        for time_vars in Rpi1
        for bandit_vars in time_vars
        for var in bandit_vars
    ] + [
        var
        for time_vars in Rpi0
        for bandit_vars in time_vars
        for var in bandit_vars
    ]
    cplex.variables.add(names=Rall_vars, lb=[0] * len(Rall_vars))

    # Objective function (maximize reward)
    Robjective = [
        (f"Rpi1_{t}_{j}_{k}", reward[j][k])
        for k in range(number_of_states)
        for j in range(number_of_bandit_in_total)
        for t in range(T0, number_of_timeperiods)
    ]
    cplex.objective.set_linear(Robjective)

    # Initial constraints
    initial_constraints = [
        SparsePair(
            ind=[
                "Rpi1_{}_{}_{}".format(T0, j, k),
                "Rpi0_{}_{}_{}".format(T0, j, k),
            ],
            val=[1, 1],
        )
        for j in range(number_of_bandit_in_total)
        for k in range(number_of_states)
    ]
    cplex.linear_constraints.add(
        lin_expr=initial_constraints,
        senses=["E"] * len(initial_constraints),
        rhs=init_prob.flatten().tolist(),
    )

    # results of minimum over p
    result_pull = np.zeros(
        (
            number_of_timeperiods,
            number_of_bandit_in_total,
            number_of_states,
            number_of_states,
        )
    )
    result_donothing = np.zeros(
        (
            number_of_timeperiods,
            number_of_bandit_in_total,
            number_of_states,
            number_of_states,
        )
    )
    for t in range(number_of_timeperiods - 1):
        for j in range(number_of_bandit_in_total):
            for k in range(number_of_states):
                for l in range(number_of_states):
                    tmp_ub_l = [
                        (
                            min(prob_pull[j][k][_] + radius_pull[t][j][k], 1),
                            min(
                                prob_donothing[j][k][_]
                                + radius_donothing[t][j][k],
                                1,
                            ),
                        )
                        for _ in range(number_of_states)
                        if _ != l
                    ]
                    # result_pull[t][j][k][l] = min(
                    #     max(
                    #         0,
                    #         prob_pull[j][k][l] - radius_pull[t][j][k],
                    #         1 - sum([t[0] for t in tmp_ub_l]),
                    #     ),
                    #     max(0, prob_pull[j][k][l] - radius_pull[t][j][k]),
                    # )
                    # result_donothing[t][j][k][l] = min(
                    #     max(
                    #         0,
                    #         prob_donothing[j][k][l] - radius_donothing[t][j][k],
                    #         1 - sum([t[1] for t in tmp_ub_l]),
                    #     ),
                    #     max(
                    #         0,
                    #         prob_donothing[j][k][l] - radius_donothing[t][j][k],
                    #     ),
                    # )
                    result_pull[t][j][k][l] = max(
                            0,
                            prob_pull[j][k][l] - radius_pull[t][j][k],
                            1 - sum([t[0] for t in tmp_ub_l]),
                        )
                    result_donothing[t][j][k][l] = max(
                            0,
                            prob_donothing[j][k][l] - radius_donothing[t][j][k],
                            1 - sum([t[1] for t in tmp_ub_l]),
                        )
    # print(result_donothing[0]==prob_donothing)
    result_pull[np.sum(result_pull, axis=3) > 1.1] = [0] * number_of_states
    result_donothing[np.sum(result_donothing, axis=3) > 1.1] = [0] * number_of_states
    
    # Time period constraints
    constraints = []
    senses = []
    rhs = []

    for t in range(T0, number_of_timeperiods):

        # Add the constraints: sum_{j}sum_{k}(Rpi1_{t,j,k} * (1-pulled_ratio * number_of_bandit_in_total)- Rpi0_{t,j,k} * pulled_ratio * number_of_bandit_in_total <=0
        curr_LHS = ["Rpi1_{}_{}_{}".format(t, j, k) for j in range(number_of_bandit_in_total) for k in range(number_of_states)] + \
                ["Rpi0_{}_{}_{}".format(t, j, k) for j in range(number_of_bandit_in_total) for k in range(number_of_states)]
        constraints.append(SparsePair(ind=curr_LHS, val=[1-pulled_ratio * change] * (len(curr_LHS)//2) + [-pulled_ratio * change] * (len(curr_LHS)//2)))
        senses.append("L")
        rhs.append(pulled_ratio * number_of_bandit_in_total * (1-change))

        if t < number_of_timeperiods - 1:
            for j in range(number_of_bandit_in_total):
                for k in range(number_of_states):
                    curr_RHS = (
                        [
                            "Rpi1_{}_{}_{}".format(t + 1, j, k),
                            "Rpi0_{}_{}_{}".format(t + 1, j, k),
                        ]
                        + [
                            "Rpi1_{}_{}_{}".format(t, j, l)
                            for l in range(number_of_states)
                        ]
                        + [
                            "Rpi0_{}_{}_{}".format(t, j, l)
                            for l in range(number_of_states)
                        ]
                    )
                    curr_RHS_coeff = (
                        [-1, -1]
                        + [
                            float(result_pull[t][j][l][k])
                            for l in range(number_of_states)
                        ]
                        + [
                            float(result_donothing[t][j][l][k])
                            for l in range(number_of_states)
                        ]
                    )
                    constraints.append(
                        SparsePair(ind=curr_RHS, val=curr_RHS_coeff)
                    )
                    senses.append("E")
                    rhs.append(0)

    cplex.linear_constraints.add(lin_expr=constraints, senses=senses, rhs=rhs)

    # Solve the problem
    cplex.parameters.timelimit.set(100)  # Set a time limit if needed
    try:
        cplex.solve()
    except CplexError as exc:
        print(exc)
        return None

    # Retrieve the objective value
    Rpi1_LP = np.zeros(
        (number_of_timeperiods, number_of_bandit_in_total, number_of_states)
    )
    Rpi0_LP = np.zeros(
        (number_of_timeperiods, number_of_bandit_in_total, number_of_states)
    )
    Robjval = cplex.solution.get_objective_value()

    # Retrieve the objective value
    objval = cplex.solution.get_objective_value()

    Rpi = (
        np.array(cplex.solution.get_values())
        .view()
        .reshape(
            2,
            number_of_timeperiods - T0,
            number_of_bandit_in_total,
            number_of_states,
        )
    )
    Rpi1_LP = Rpi[0, :, :, :]
    Rpi0_LP = Rpi[1, :, :, :]

    # # Write the objective value to the output file
    # fout.write(f"{Robjval}\n")

    return Robjval, Rpi1_LP, Rpi0_LP


def monte_carlo_randomized_policy_infeasible(
    pi1_LP,
    pi0_LP,
    number_of_timeperiods,
    number_of_states,
    reward,
    prob_pull,
    pulled_ratio,
    number_of_bandit_in_total,
    init_prob,
):
    """
    Simulate a Monte Carlo randomized policy for an infeasible solution.

    Args:
    pi1_LP (list): A 3D list of pi values from the LP solution.
    pi0_LP (list): A 3D list of pi values from the LP solution.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 3D list of rewards of pull action.
    prob_pull (list): A 4D list of transition probabilities of pull action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.
    init_prob (list): A 2D list of initial probabilities.

    Returns:
    float: The total reward obtained from the simulation.
    """
    total_reward = 0.0
    active_bandit_number = 0
    
    # Initialization of state for each arm based on the initial probability
    arm_state = np.array(
        [
            np.random.choice(number_of_states, p=init_prob[j])
            for j in range(number_of_bandit_in_total)
        ]
    )

    # Vectorized simulation for each time period
    for t in range(number_of_timeperiods):
        # Extract current pi1 and pi0 values for the current time period
        pi1_t = pi1_LP[t, np.arange(number_of_bandit_in_total), arm_state]
        pi0_t = pi0_LP[t, np.arange(number_of_bandit_in_total), arm_state]

        # Compute active probabilities for all bandits (vectorized)
        sum_pi = pi1_t + pi0_t

        active_probs = np.divide(
            pi1_t, sum_pi, out=np.zeros_like(pi1_t), where=sum_pi > 0
        )

        # Randomly determine which arms are active based on the active probabilities (vectorized)
        random_values = np.random.rand(number_of_bandit_in_total)
        active = random_values < active_probs

        # Calculate the weights (1 + active_prob if active, else 0)
        action = np.where(active, 1, 0)
        state_idx = np.array(
            [i for i in range(number_of_bandit_in_total) if action[i] > 0]
        )

        # Update total reward and arm states based on actions (vectorized)
        reward_for_pulled = (
            reward[np.arange(number_of_bandit_in_total), arm_state] * action
        )
        total_reward += np.sum(reward_for_pulled)

        # Update the state of pulled arms based on transition probabilities (vectorized)
        if len(state_idx) > 0:
            arm_state[state_idx] = generate_number_in_dist_batch(
                prob_pull[state_idx, arm_state[state_idx], :]
            )
        active_bandit_number += np.sum(action)
        # print(f"Allowed bandit number: {int(pulled_ratio * number_of_bandit_in_total)}")
        print(f"Number of active bandits of fluid LP: {np.sum(action)}")

    return total_reward, active_bandit_number/number_of_timeperiods


def monte_carlo_randomized_policy_feasible(
    pi1_LP,
    pi0_LP,
    number_of_timeperiods,
    number_of_states,
    reward,
    prob_pull,
    pulled_ratio,
    number_of_bandit_in_total,
    init_prob,
):
    """
    Simulate a Monte Carlo randomized policy for a feasible solution.

    Args:
        pi1_LP (list): A 3D list of pi values from the LP solution.
        pi0_LP (list): A 3D list of pi values from the LP solution.
        number_of_timeperiods (int): The number of time periods.
        number_of_states (int): The number of states.
        reward (list): A 3D list of rewards of pull action.
        prob_pull (list): A 4D list of transition probabilities of pull action.
        pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
        number_of_bandit_in_total (int): The total number of bandits.
        init_prob (list): A 2D list of initial probabilities.

    Returns:
        float: The total reward obtained from the simulation.
    """

    total_reward = 0.0
    active_bandit_number = 0
    
    # Generate initial states for all arms based on their initial probabilities (vectorized)
    arm_state = np.array(
        [
            np.random.choice(number_of_states, p=init_prob[j])
            for j in range(number_of_bandit_in_total)
        ]
    )
    # Vectorized simulation for each time period
    for t in range(number_of_timeperiods):
        # Extract current pi1 and pi0 values for the current time period
        pi1_t = pi1_LP[t, np.arange(number_of_bandit_in_total), arm_state]
        pi0_t = pi0_LP[t, np.arange(number_of_bandit_in_total), arm_state]

        # Compute active probabilities for all bandits (vectorized)
        sum_pi = pi1_t + pi0_t
        active_probs = np.divide(
            pi1_t, sum_pi, out=np.zeros_like(pi1_t), where=sum_pi > 0
        )

        # Randomly determine which arms are active based on the active probabilities (vectorized)
        random_values = np.random.rand(number_of_bandit_in_total)
        active = random_values < active_probs

        # Calculate the weights (1 + active_prob if active, else 0)
        weights = np.where(active, 1 + active_probs, 0)

        # Sort and select top arms based on the weights, respecting the pulled_ratio constraint
        action, num_active = sort_truncated(
            weights,
            number_of_bandit_in_total,
            int(pulled_ratio * number_of_bandit_in_total),
        )
        state_idx = np.array(
            [i for i in range(number_of_bandit_in_total) if action[i] > 0]
        )

        # Update total reward and arm states based on actions (vectorized)
        reward_for_pulled = (
            reward[np.arange(number_of_bandit_in_total), arm_state] * action
        )
        total_reward += np.sum(reward_for_pulled)

        # Update the state of pulled arms based on transition probabilities (vectorized)
        if len(state_idx) > 0:
            arm_state[state_idx] = generate_number_in_dist_batch(
                prob_pull[state_idx, arm_state[state_idx], :]
            )
        active_bandit_number += np.sum(action)
        
    return total_reward, active_bandit_number/number_of_timeperiods


def generate_sample_paths(
    init_prob,
    prob_pull,
    prob_donothing,
    reward,
    number_of_bandit_in_total,
    number_of_states,
    path_length,
    num_paths,
    pulled_ratio,
    selecting_rule=0.6,
):
    """
    Generate sample paths for the bandit problem.

    Args:
    init_prob (np.array): Initial probabilities for each bandit.
    prob_pull (np.array): Transition probabilities for the pull action.
    prob_donothing (list): Transition probabilities for the do-nothing action.
    reward (np.array): Reward function.
    number_of_bandit_in_total (int): The total number of bandits.
    number_of_states (int): The number of states.
    path_length (int): The length of each path.
    num_paths (int): The number of sample paths to generate.

    Returns:
    paths (list): A list of sample paths.
    """

    path = []

    for _ in range(num_paths):
        trajectory = []
        arm_state = np.array(
            [
                np.random.choice(number_of_states, p=init_prob[j])
                for j in range(number_of_bandit_in_total)
            ]
        )
        for _ in range(path_length):
            # Randomly determine which arms are active based on the active probabilities (vectorized)
            random_values = np.random.rand(number_of_bandit_in_total)
            active = random_values < selecting_rule

            # Calculate the weights (1 + active_prob if active, else 0)
            weights = np.where(active, 1, 0)

            # Sort and select top arms based on the weights, respecting the pulled_ratio constraint
            action, num_active = random_truncated(
                weights,
                number_of_bandit_in_total,
                int(pulled_ratio * number_of_bandit_in_total),
            )
            state_idx = np.array(
                [i for i in range(number_of_bandit_in_total) if action[i] > 0]
            )

            # Update total reward and arm states based on actions (vectorized)
            reward_for_pulled = (
                reward[np.arange(number_of_bandit_in_total), arm_state] * action
            )
            # trajectory.append(list(zip(arm_state.copy(), action, reward_for_pulled.copy())))
            trajectory.append(
                np.column_stack(
                    (arm_state.copy(), action, reward_for_pulled.copy())
                )
            )
            # Update the state of pulled arms based on transition probabilities (vectorized)
            if len(state_idx) > 0:
                arm_state[state_idx] = generate_number_in_dist_batch(
                    prob_pull[state_idx, arm_state[state_idx], :]
                )
        path.append(trajectory)
    # print(np.array(path).shape)
    path = np.array(
        [np.array(path)[:, :, i, :] for i in range(number_of_bandit_in_total)]
    )
    return path


def calculate_mle_transition_count(
    sample_path,
    number_of_bandit_in_total,
    number_of_states,
    path_length,
    number_of_actions=2,
):
    # Initialize dictionaries to count transitions and state-action occurrences
    N_sa = defaultdict(int)  # Count N^{a_i}_{s_i}
    N_sas = defaultdict(int)  # Count N^{a_i}_{s_i,s_i'}
    initial_prob = np.zeros(
        (number_of_bandit_in_total, number_of_states)
    )  # Initial state probability distribution

    # Loop over each bandit and its sample paths
    for i in range(number_of_bandit_in_total):
        for path in sample_path[i]:
            # Increment initial state count
            initial_state = int(path[0][0])  # First state of the path
            initial_prob[i, initial_state] += 1

            # Loop over the time steps in the path (excluding the last step)
            for t in range(path_length - 1):
                s_t = int(path[t][0])  # Current state
                a_t = int(path[t][1])  # Current action
                s_tp1 = int(path[t + 1][0])  # Next state

                # Increment counts for the current state-action pair and the state-action-next-state triple
                N_sa[(i, s_t, a_t)] += 1
                N_sas[(i, s_t, a_t, s_tp1)] += 1

    # Normalize initial state probabilities
    initial_prob /= initial_prob.sum(axis=1, keepdims=True)
    return N_sa, N_sas, initial_prob


# Calculate MLE transition kernel
def count_to_kernel(
    N_sa,
    N_sas,
    number_of_bandit_in_total,
    number_of_states,
    number_of_actions=2,
):
    transition_kernel = np.zeros(
        (
            number_of_bandit_in_total,
            number_of_states,
            number_of_actions,
            number_of_states,
        )
    )
    for (i, s_t, a_t, s_tp1), count in N_sas.items():
        N_sa_value = N_sa[(i, s_t, a_t)]
        if N_sa_value > 0:
            transition_kernel[i, s_t, a_t, s_tp1] = count / N_sa_value
    return transition_kernel[:, :, 1, :], transition_kernel[:, :, 0, :]


def monte_carlo_randomized_policy_infeasible_online(
    number_of_timeperiods,
    number_of_states,
    reward,
    prob_pull,
    prob_donothing,
    prob_pull_mle,
    prob_donothing_mle,
    pulled_ratio,
    number_of_bandit_in_total,
    init_prob,
    init_prob_mle,
    N_sa,
    N_sas,
    function_obtain_policy,
    radius_pull,
    radius_donothing,
    mixed,
    change,
):
    """
    Simulate a Monte Carlo randomized policy for an infeasible solution.

    Args:
    pi1_LP (list): A 3D list of pi values from the LP solution.
    pi0_LP (list): A 3D list of pi values from the LP solution.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 2D list of rewards of pull action.
    prob_pull (list): A 3D list of transition probabilities of pull action.
    prob_pull (list): A 3D list of transition probabilities of donothing action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.
    init_prob (list): A 2D list of initial probabilities.

    Returns:
    float: The total reward obtained from the simulation.
    """
    total_reward = 0.0
    active_bandit_number = 0

    # Initialization of state for each arm based on the initial probability
    arm_state = np.array(
        [
            np.random.choice(number_of_states, p=init_prob_mle[j])
            for j in range(number_of_bandit_in_total)
        ]
    )
    _, pi1_LP0, pi0_LP0 = function_obtain_policy(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        init_prob_mle,
        prob_pull_mle,
        prob_donothing_mle,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change,
    )
    # Vectorized simulation for each time period
    for t in range(number_of_timeperiods):
        # Extract current pi1 and pi0 values for the current time period
        _, pi1_LP, pi0_LP = function_obtain_policy(
            t,
            number_of_timeperiods,
            number_of_states,
            reward,
            init_prob_mle,
            prob_pull_mle,
            prob_donothing_mle,
            pulled_ratio,
            number_of_bandit_in_total,
            radius_pull,
            radius_donothing,
            change,
        )
        pi1_t = (
            pi1_LP[0, np.arange(number_of_bandit_in_total), arm_state]
            + mixed
            * pi1_LP0[t, np.arange(number_of_bandit_in_total), arm_state]
        )
        pi0_t = (
            pi0_LP[0, np.arange(number_of_bandit_in_total), arm_state]
            + mixed
            * pi0_LP0[t, np.arange(number_of_bandit_in_total), arm_state]
        )

        # Compute active probabilities for all bandits (vectorized)
        sum_pi = pi1_t + pi0_t
        active_probs = np.where(
            sum_pi > 0,
            np.divide(
                pi1_t, sum_pi, out=np.zeros_like(pi1_t), where=sum_pi > 0
            ),
            0,
        )
        # print(active_probs)

        # Randomly determine which arms are active based on the active probabilities (vectorized)
        random_values = np.random.rand(number_of_bandit_in_total)
        active = random_values < active_probs

        # Calculate the weights (1 + active_prob if active, else 0)
        action = np.where(active, 1, 0)
        state_idx = np.where(action > 0)[0]
        state_idx_donothing = np.where(action == 0)[0]
        arm_state_copy = arm_state.copy()

        # Update total reward and arm states based on actions (vectorized)
        reward_for_pulled = (
            reward[np.arange(number_of_bandit_in_total), arm_state_copy]
            * action
        )
        total_reward += np.sum(reward_for_pulled)

        # Update the state of pulled arms based on transition probabilities (vectorized)
        if state_idx.size > 0:
            arm_state[state_idx] = generate_number_in_dist_batch(
                prob_pull[state_idx, arm_state[state_idx], :]
            )

        # print(arm_state == arm_state_copy)
        for i in range(number_of_bandit_in_total):
            s_i = arm_state_copy[i]  # Current state
            a_i = action[i]  # Current action
            s_ip1 = arm_state[i]  # Next state
            N_sa[(i, s_i, a_i)] += 1
            N_sas[(i, s_i, a_i, s_ip1)] += 1

        init_prob = np.zeros_like(init_prob)
        init_prob[np.arange(number_of_bandit_in_total), arm_state] = 1
        active_bandit_number += np.sum(action)
        # print(f"Allowed bandit number: {int(pulled_ratio * number_of_bandit_in_total)}")
        # print(f"Number of active bandits of fluid LP: {np.sum(action)}")

    return total_reward, active_bandit_number / number_of_timeperiods


def monte_carlo_randomized_policy_feasible_online(
    number_of_timeperiods,
    number_of_states,
    reward,
    prob_pull,
    prob_donothing,
    prob_pull_mle,
    prob_donothing_mle,
    pulled_ratio,
    number_of_bandit_in_total,
    init_prob,
    init_prob_mle,
    N_sa,
    N_sas,
    function_obtain_policy,
    radius_pull,
    radius_donothing,
    mixed,
    change,
):
    """
    Simulate a Monte Carlo randomized policy for an infeasible solution.

    Args:
    pi1_LP (list): A 3D list of pi values from the LP solution.
    pi0_LP (list): A 3D list of pi values from the LP solution.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 2D list of rewards of pull action.
    prob_pull (list): A 3D list of transition probabilities of pull action.
    prob_pull (list): A 3D list of transition probabilities of donothing action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.
    init_prob (list): A 2D list of initial probabilities.

    Returns:
    float: The total reward obtained from the simulation.
    """
    total_reward = 0.0
    active_bandit_number = 0
    
    # Initialization of state for each arm based on the initial probability
    arm_state = np.array(
        [
            np.random.choice(number_of_states, p=init_prob_mle[j])
            for j in range(number_of_bandit_in_total)
        ]
    )
    _, pi1_LP0, pi0_LP0 = function_obtain_policy(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        init_prob_mle,
        prob_pull_mle,
        prob_donothing_mle,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change,
    )
    # Vectorized simulation for each time period
    for t in range(number_of_timeperiods):
        # Extract current pi1 and pi0 values for the current time period
        _, pi1_LP, pi0_LP = function_obtain_policy(
            t,
            number_of_timeperiods,
            number_of_states,
            reward,
            init_prob_mle,
            prob_pull_mle,
            prob_donothing_mle,
            pulled_ratio,
            number_of_bandit_in_total,
            radius_pull,
            radius_donothing,
            change,
        )
        pi1_t = (
            pi1_LP[0, np.arange(number_of_bandit_in_total), arm_state]
            + mixed
            * pi1_LP0[t, np.arange(number_of_bandit_in_total), arm_state]
        )
        pi0_t = (
            pi0_LP[0, np.arange(number_of_bandit_in_total), arm_state]
            + mixed
            * pi0_LP0[t, np.arange(number_of_bandit_in_total), arm_state]
        )

        # Compute active probabilities for all bandits (vectorized)
        sum_pi = pi1_t + pi0_t
        active_probs = np.where(
            sum_pi > 0,
            np.divide(
                pi1_t, sum_pi, out=np.zeros_like(pi1_t), where=sum_pi > 0
            ),
            0,
        )
        # print(active_probs)

        # Randomly determine which arms are active based on the active probabilities (vectorized)
        random_values = np.random.rand(number_of_bandit_in_total)
        active = random_values < active_probs

        # Calculate the weights (1 + active_prob if active, else 0)
        weights = np.where(active, 1 + active_probs, 0)

        # Sort and select top arms based on the weights, respecting the pulled_ratio constraint
        action, num_active = sort_truncated(
            weights,
            number_of_bandit_in_total,
            int(pulled_ratio * number_of_bandit_in_total),
        )
        state_idx = np.array(
            [i for i in range(number_of_bandit_in_total) if action[i] > 0]
        )
        arm_state_copy = arm_state.copy()

        # Update total reward and arm states based on actions (vectorized)
        reward_for_pulled = (
            reward[np.arange(number_of_bandit_in_total), arm_state] * action
        )
        total_reward += np.sum(reward_for_pulled)

        # Update the state of pulled arms based on transition probabilities (vectorized)
        if len(state_idx) > 0:
            arm_state[state_idx] = generate_number_in_dist_batch(
                prob_pull[state_idx, arm_state[state_idx], :]
            )

        # print(arm_state == arm_state_copy)
        for i in range(number_of_bandit_in_total):
            s_i = arm_state_copy[i]  # Current state
            a_i = action[i]  # Current action
            s_ip1 = arm_state[i]  # Next state
            N_sa[(i, s_i, a_i)] += 1
            N_sas[(i, s_i, a_i, s_ip1)] += 1

        init_prob = np.zeros_like(init_prob)
        init_prob[np.arange(number_of_bandit_in_total), arm_state] = 1
        
        active_bandit_number += np.sum(action)
        # print(f"Allowed bandit number: {int(pulled_ratio * number_of_bandit_in_total)}")
        # print(f"Number of active bandits of fluid LP: {np.sum(action)}")

    return total_reward, active_bandit_number / number_of_timeperiods


def policy_ratio(
    number_of_timeperiods,
    number_of_bandit_in_total,
    number_of_states,
    init_prob,
    prob_pull,
    prob_donothing,
    reward,
    initial_prob,
    transition_kernel_pull,
    transition_kernel_donothing,
    pulled_ratio,
    radius_pull,
    radius_donothing,
    N_sa,
    N_sas,
):

    # print("\n*****************Fluid LP *******************************")
    objval_true, pi1_LP_true, pi0_LP_true = fluid_LP(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        init_prob,
        prob_pull,
        prob_donothing,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change = None,
    )
    objval_mle, pi1_LP_mle, pi0_LP_mle = fluid_LP(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        initial_prob,
        transition_kernel_pull,
        transition_kernel_donothing,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change = None,
    )

    infeasible_total_reward_true, _ = monte_carlo_randomized_policy_infeasible(
        pi1_LP_true,
        pi0_LP_true,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )
    infeasible_total_reward_mle, infeasible_active_bandit_number_mle = monte_carlo_randomized_policy_infeasible(
        pi1_LP_mle,
        pi0_LP_mle,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )

    # feasible_total_reward_true = monte_carlo_randomized_policy_feasible(
    #     pi1_LP_true,
    #     pi0_LP_true,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     prob_pull,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     init_prob,
    # )
    feasible_total_reward_mle, feasible_active_bandit_number_mle = monte_carlo_randomized_policy_feasible(
        pi1_LP_mle,
        pi0_LP_mle,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )

    # print("\n*****************Constant Decision rule *******************************")
    # Robjval_true, Rpi1_LP_true, Rpi0_LP_true = constant_decision_rule_LP(
    #     0,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     init_prob,
    #     prob_pull,
    #     prob_donothing,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     radius_pull,
    #     radius_donothing,
    #     change = 0,
    # )
    Robjval_mle_noc_b, Rpi1_LP_mle_noc_b, Rpi0_LP_mle_noc_b = constant_decision_rule_LP(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        initial_prob,
        transition_kernel_pull,
        transition_kernel_donothing,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change = 0,
    )

    # infeasible_Rtotal_reward_true = monte_carlo_randomized_policy_infeasible(
    #     Rpi1_LP_true,
    #     Rpi0_LP_true,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     prob_pull,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     init_prob,
    # )
    infeasible_Rtotal_reward_mle_noc_b, infeasible_active_bandit_number_Rmle_noc_b = monte_carlo_randomized_policy_infeasible(
        Rpi1_LP_mle_noc_b,
        Rpi0_LP_mle_noc_b,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )

    # feasible_Rtotal_reward_true = monte_carlo_randomized_policy_feasible(
    #     Rpi1_LP_true,
    #     Rpi0_LP_true,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     prob_pull,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     init_prob,
    # )
    feasible_Rtotal_reward_mle_noc_b, feasible_active_bandit_number_Rmle_noc_b = monte_carlo_randomized_policy_feasible(
        Rpi1_LP_mle_noc_b,
        Rpi0_LP_mle_noc_b,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )
    
    Robjval_mle_c_b, Rpi1_LP_mle_c_b, Rpi0_LP_mle_c_b = constant_decision_rule_LP(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        initial_prob,
        transition_kernel_pull,
        transition_kernel_donothing,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change = 1,
    )
    
    infeasible_Rtotal_reward_mle_c_b, infeasible_active_bandit_number_Rmle_c_b = monte_carlo_randomized_policy_infeasible(
        Rpi1_LP_mle_c_b,
        Rpi0_LP_mle_c_b,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )   
    
    feasible_Rtotal_reward_mle_c_b, feasible_active_bandit_number_Rmle_c_b = monte_carlo_randomized_policy_feasible(
        Rpi1_LP_mle_c_b,
        Rpi0_LP_mle_c_b,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )    
    
    
    # print("\n***************** Online Policy *******************************")

    infeasible_total_reward_mle_online_unmixed, infeasible_active_bandit_number_mle_online_unmixed = (
        monte_carlo_randomized_policy_infeasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            fluid_LP,
            radius_pull,
            radius_donothing,
            mixed = 0,
            change = 1,
        )
    )
    feasible_total_reward_mle_online_unmixed, feasible_active_bandit_number_mle_online_unmixed = (
        monte_carlo_randomized_policy_feasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            fluid_LP,
            radius_pull,
            radius_donothing,
            mixed = 0,
            change = 1,
        )
    )
    infeasible_total_reward_Rmle_online_unmixed, infeasible_active_bandit_number_Rmle_online_unmixed = (
        monte_carlo_randomized_policy_infeasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            constant_decision_rule_LP,
            radius_pull,
            radius_donothing,
            mixed = 0,
            change = 1,
        )
    )
    feasible_total_reward_Rmle_online_unmixed, feasible_active_bandit_number_Rmle_online_unmixed = (
        monte_carlo_randomized_policy_feasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            constant_decision_rule_LP,
            radius_pull,
            radius_donothing,
            mixed = 0,
            change = 1,
        )
    )

    infeasible_total_reward_mle_online_mixed, infeasible_active_bandit_number_mle_online_mixed = (
        monte_carlo_randomized_policy_infeasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            fluid_LP,
            radius_pull,
            radius_donothing,
            mixed = 1,
            change = 0,
        )
    )
    feasible_total_reward_mle_online_mixed, feasible_active_bandit_number_mle_online_mixed = (
        monte_carlo_randomized_policy_feasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            fluid_LP,
            radius_pull,
            radius_donothing,
            mixed = 1,
            change = 0,
        )
    )
    infeasible_total_reward_Rmle_online_mixed, infeasible_active_bandit_number_Rmle_online_mixed = (
        monte_carlo_randomized_policy_infeasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            constant_decision_rule_LP,
            radius_pull,
            radius_donothing,
            mixed = 1,
            change = 0,
        )
    )
    feasible_total_reward_Rmle_online_mixed, feasible_active_bandit_number_Rmle_online_mixed = (
        monte_carlo_randomized_policy_feasible_online(
            number_of_timeperiods,
            number_of_states,
            reward,
            prob_pull,
            prob_donothing,
            transition_kernel_pull,
            transition_kernel_donothing,
            pulled_ratio,
            number_of_bandit_in_total,
            init_prob,
            initial_prob,
            N_sa,
            N_sas,
            constant_decision_rule_LP,
            radius_pull,
            radius_donothing,
            mixed = 1,
            change = 0,
        )
    )
    return np.array(
        [
            infeasible_total_reward_mle / infeasible_total_reward_true,
            feasible_total_reward_mle / infeasible_total_reward_true,
            infeasible_Rtotal_reward_mle_noc_b / infeasible_total_reward_true,
            feasible_Rtotal_reward_mle_noc_b / infeasible_total_reward_true,
            infeasible_Rtotal_reward_mle_c_b / infeasible_total_reward_true,
            feasible_Rtotal_reward_mle_c_b / infeasible_total_reward_true,           
            infeasible_total_reward_mle_online_unmixed
            / infeasible_total_reward_true,
            feasible_total_reward_mle_online_unmixed
            / infeasible_total_reward_true,
            infeasible_total_reward_Rmle_online_unmixed
            / infeasible_total_reward_true,
            feasible_total_reward_Rmle_online_unmixed
            / infeasible_total_reward_true,
            infeasible_total_reward_mle_online_mixed
            / infeasible_total_reward_true,
            feasible_total_reward_mle_online_mixed
            / infeasible_total_reward_true,
            infeasible_total_reward_Rmle_online_mixed
            / infeasible_total_reward_true,
            feasible_total_reward_Rmle_online_mixed
            / infeasible_total_reward_true,
        ]
    ), np.array(
        [
            infeasible_active_bandit_number_mle, 
            feasible_active_bandit_number_mle, 
            infeasible_active_bandit_number_Rmle_noc_b, 
            feasible_active_bandit_number_Rmle_noc_b, 
            infeasible_active_bandit_number_Rmle_c_b, 
            feasible_active_bandit_number_Rmle_c_b, 
            infeasible_active_bandit_number_mle_online_unmixed, 
            feasible_active_bandit_number_mle_online_unmixed, 
            infeasible_active_bandit_number_Rmle_online_unmixed, 
            feasible_active_bandit_number_Rmle_online_unmixed, 
            infeasible_active_bandit_number_mle_online_mixed, 
            feasible_active_bandit_number_mle_online_mixed, 
            infeasible_active_bandit_number_Rmle_online_mixed, 
            feasible_active_bandit_number_Rmle_online_mixed
                
        ]
        )
    # return np.array([objval_true/Robjval_true, objval_mle/Robjval_mle])


def get_single_sample(
    sample_idx: int,
    number_of_timeperiods: int,
    number_of_states: int,
    number_of_bandit_in_total: int,
    path_length: int,
    num_paths: int,
    pulled_ratio: float,
    r: float,
    selecting_rule: float = 0.6,
) -> Tuple[Tuple, Tuple]:
    random.seed(int(sample_idx * 10))
    np.random.seed(int(sample_idx * 10))
    (
        init_prob,
        prob_pull,
        prob_donothing,
        reward,
        radius_pull,
        radius_donothing,
    ) = initialization(
        number_of_bandit_in_total,
        number_of_states,
        number_of_timeperiods,
        r,
    )
    sample_paths = generate_sample_paths(
        init_prob,
        prob_pull,
        prob_donothing,
        reward,
        number_of_bandit_in_total,
        number_of_states,
        path_length,
        num_paths,
        pulled_ratio,
        selecting_rule,
    )
    N_sa, N_sas, initial_prob = calculate_mle_transition_count(
        sample_paths,
        number_of_bandit_in_total,
        number_of_states,
        path_length,
        number_of_actions=2,
    )
    transition_kernel_pull, transition_kernel_donothing = count_to_kernel(
        N_sa,
        N_sas,
        number_of_bandit_in_total,
        number_of_states,
        number_of_actions=2,
    )
    return (
        (
            init_prob,
            prob_pull,
            prob_donothing,
            reward,
            radius_pull,
            radius_donothing,
        ),
        (
            initial_prob,
            transition_kernel_pull,
            transition_kernel_donothing,
            N_sa,
            N_sas,
        ),
    )
    
def RLagrangian(prob_true, value_last_time, radius, number_of_states):
    """
    Solve the constant_decision_rule_LP using CPLEX.

    Args:
        prob_true (list): 1D list of transition probabilities (true probabilities).
        value_last_time (list): Coefficients for the objective function.
        radius (float): Radius for the ambiguity set.
        number_of_states (int): The number of states.

    Returns:
        float: The objective value of the solved LP.
        list: Optimal solution for the Rprob variables.
    """
    # Initialize CPLEX environment and model
    cplex = Cplex()
    cplex.set_problem_type(Cplex.problem_type.LP)
    cplex.objective.set_sense(cplex.objective.sense.maximize)

    # Define Rprob variables
    Rprob = [f"Rprob_{i}" for i in range(number_of_states)]
    cplex.variables.add(names=Rprob, lb=[0] * number_of_states)

    # Set objective function (maximize value_last_time * Rprob)
    objective_terms = [(f"Rprob_{i}", value_last_time[i]) for i in range(number_of_states)]
    cplex.objective.set_linear(objective_terms)

    # Add constraint: sum(Rprob) == 1 (probabilities must sum to 1)
    cplex.linear_constraints.add(
        lin_expr=[SparsePair(ind=Rprob, val=[1] * number_of_states)],
        senses=["E"],
        rhs=[1]
    )

    # Add ambiguity set constraints: radius constraints
    for i in range(number_of_states):
        # Lower bound: Rprob[i] >= prob_true[i] - radius
        cplex.linear_constraints.add(
            lin_expr=[SparsePair(ind=[f"Rprob_{i}"], val=[1])],
            senses=["G"],
            rhs=[prob_true[i] - radius]
        )
        # Upper bound: Rprob[i] <= prob_true[i] + radius
        cplex.linear_constraints.add(
            lin_expr=[SparsePair(ind=[f"Rprob_{i}"], val=[1])],
            senses=["L"],
            rhs=[prob_true[i] + radius]
        )

    # Set time limit for solving
    cplex.parameters.timelimit.set(100)

    # Solve the problem
    try:
        cplex.solve()
    except CplexError as exc:
        print(f"CPLEX error: {exc}")
        return None

    # Retrieve the objective value
    RLobjval = cplex.solution.get_objective_value()

    # Retrieve the optimal values for Rprob
    Rprob_return = [cplex.solution.get_values(f"Rprob_{i}") for i in range(number_of_states)]

    return RLobjval, Rprob_return

def RLagrangian_unwrapper(args: Tuple):
    return RLagrangian(*args)


def parallel_solve_LPs(lp_problems, number_of_states, num_proc):
    """
    Solve multiple LPs in parallel.
    
    Args:
    lp_problems (list): List of problem instances (each a tuple with prob_true, value_last_time, radius).
    number_of_states (int): Number of states.
    
    Returns:
    A list of results for each LP.
    """
    results = []
    
    # Using ThreadPoolExecutor for parallel processing

    with mp.Pool(processes=num_proc) as pool:
        results = list(
            tqdm(
                pool.imap(RLagrangian_unwrapper, lp_problems),
                total=len(lp_problems),
                desc=f"Calculating policy ratio with poolsize={num_proc}",
            )
        )
    
    return results

def RLagrangian_relaxation_parallel(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, mu, T0, num_proc):
    """
    Solve the robust Lagrangian relaxation using backward induction and CPLEX, with parallelization for LP problems.

    Args:
        number_of_timeperiods (int): The number of time periods.
        number_of_states (int): The number of states.
        reward (list): A 3D list of rewards for the pull action.
        init_prob (list): A 2D list of initial probabilities.
        prob_pull (list): A 4D list of transition probabilities for pull action.
        prob_donothing (list): A 4D list of transition probabilities for do-nothing action.
        pulled_ratio (float): The maximum percentage of bandits that can be pulled.
        number_of_bandit_in_total (int): Total number of bandits.
        radius_pull (list): Radius of ambiguity set for pull action.
        radius_donothing (list): Radius of ambiguity set for do-nothing action.
        mu (float): Lagrangian multiplier.
        T0 (int): End time of backward induction.

    Returns:
        tuple: (value, action, RLgradient_T0, Rprob_pull, Rprob_donothing, RLvalue_T0)
    """
    
    # Initialize value, action, gradient, and Rprob arrays
    value = np.zeros((number_of_timeperiods, number_of_bandit_in_total, number_of_states))
    action = np.zeros((number_of_timeperiods, number_of_bandit_in_total, number_of_states), dtype=int)
    gradient = np.zeros((number_of_timeperiods, number_of_bandit_in_total, number_of_states))
    
    Rprob_pull_batch = np.zeros((number_of_timeperiods, number_of_bandit_in_total, number_of_states, number_of_states))
    Rprob_donothing_batch = np.zeros((number_of_timeperiods, number_of_bandit_in_total, number_of_states, number_of_states))
    
    reward_at_time = np.array(reward)
    value[number_of_timeperiods - 1] = np.where(reward_at_time > mu, reward_at_time - mu, 0)
    action[number_of_timeperiods - 1] = np.where(reward_at_time > mu, 1, 0)
    gradient[number_of_timeperiods - 1] = np.where(reward_at_time > mu, -1, 0)
    # Backward induction
    for tau in range(1, number_of_timeperiods - T0):
        time_index = number_of_timeperiods - tau - 1
        lp_problems = []  # Collect all LP problems for parallel solving

        # Prepare LP problems for parallel solving
        for j in range(number_of_bandit_in_total):
            for k in range(number_of_states):
                # Append LP problems for both pull and do-nothing actions
                lp_problems.append((prob_pull[j][k], value[time_index + 1][j], radius_pull[time_index][j][k], number_of_states))
                lp_problems.append((prob_donothing[j][k], value[time_index + 1][j], radius_donothing[time_index][j][k], number_of_states))

        # Solve all LP problems in parallel
        lp_results = parallel_solve_LPs(lp_problems, number_of_states, num_proc)
        # print(lp_results)
        value_res, Rprob_res = zip(*lp_results)
        
        pull_value_batch = np.array(value_res).view().reshape(number_of_bandit_in_total, number_of_states, 2)[:,:, 0]
        donothing_value_batch = np.array(value_res).view().reshape(number_of_bandit_in_total, number_of_states, 2)[:,:, 1]
        Rprob_pull_batch[time_index] = np.array(Rprob_res).view().reshape(number_of_bandit_in_total, number_of_states, 2, number_of_states)[:,:, 0, :]
        Rprob_donothing_batch[time_index] = np.array(Rprob_res).view().reshape(number_of_bandit_in_total, number_of_states, 2, number_of_states)[:,:, 1, :]
        
        pull_value = reward - mu + pull_value_batch
        
        # Stack pull_value and donothing_value_batch along a new axis (axis=-1)
        stacked_values = np.stack([donothing_value_batch, pull_value], axis=-1)

        # Take the max along the new axis
        value[time_index] = np.max(stacked_values, axis=-1)

        # Take the argmax along the new axis to determine the action
        action[time_index] = np.argmax(stacked_values, axis=-1)
        
        # Perform the matrix multiplication using einsum
        pull_prob = np.einsum('ijk,ik->ij', Rprob_pull_batch[time_index], gradient[time_index + 1])
        donothing_prob = np.einsum('ijk,ik->ij', Rprob_donothing_batch[time_index], gradient[time_index + 1])

        # Update gradient using matrix form
        gradient[time_index] = -action[time_index] + action[time_index] * pull_prob + (1 - action[time_index]) * donothing_prob

    # Compute weighted value and gradient at T0
    weighted_value = np.sum(np.multiply(value[T0], init_prob))
    weighted_gradient = np.sum(np.multiply(gradient[T0], init_prob))

    # Compute final RLvalue_T0 and RLgradient_T0
    RLvalue_T0 = (number_of_timeperiods - T0) * number_of_bandit_in_total * pulled_ratio * mu + weighted_value
    RLgradient_T0 = (number_of_timeperiods - T0) * number_of_bandit_in_total * pulled_ratio + weighted_gradient

    return value, action, RLgradient_T0, Rprob_pull_batch, Rprob_donothing_batch, RLvalue_T0

def subgradient(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, T0, num_proc):
    """
    Solve the constant_decision_rule_LP using CPLEX.

    Args:
    # pi1_LP (list): A 3D list to store the pi variables with pull arms.
    # pi0_LP (list): A 3D list to store the pi variables with donothing action.
    fout (file object): The output file object.
    number_of_timeperiods (int): The number of time periods.
    number_of_states (int): The number of states.
    reward (list): A 3D list of rewards of pull action.
    init_prob (list): A 2D list of initial probabilities.
    prob_pull (list): A 4D list of transition probabilities of pull action.
    prob_dothing (list): A 4D list of transition probabilities of dothing action.
    pulled_ratio (float): The maximum percentage of bandits that are allowed to be pulled.
    number_of_bandit_in_total (int): The total number of bandits.
    radius_pull (list): The list to store the radius of the ambuguity set of pull action.
    radius_donothing (list): The list to store the radius of the ambuguity set of donothing action.
    mu: Lagrangian parameter
    T0: End time of backward induction

    Returns:
    The objective value of the solved robust Lagrangian relaxation, and the optimal solutions.
    """
    iterations = 10
    learning_rate = 0.003
    stopping_threshold = 1e-6
    
    # generate some randomized parameter initial values
    mu_GD = np.zeros((iterations))
    mu_GD[0] = random.random()
    
    # set up some trackers for the training process
    costs = np.zeros((iterations))
    gradient = np.zeros((iterations))
    value = [None] * iterations
    action = [None] * iterations
    Rprob_pull_batch = [None] * iterations
    Rprob_donothing_batch = [None] * iterations
    value[0], action[0], gradient[0], Rprob_pull_batch[0], Rprob_donothing_batch[0], costs[0] = RLagrangian_relaxation_parallel(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, mu_GD[0], T0, num_proc)
    
    # training loop
    for i in tqdm(range(iterations-1)):
        # 1. use current params to get predicted output
        mu_GD[i+1] = max(mu_GD[i] - gradient[i] * learning_rate, 0)
        value[i+1], action[i+1], gradient[i+1], Rprob_pull_batch[i+1], Rprob_donothing_batch[i+1], costs[i+1] = RLagrangian_relaxation_parallel(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, mu_GD[i+1], T0, num_proc)

        # 2. check if converged and record trackers
        if (abs(costs[i+1] - costs[i]) < stopping_threshold):
            print(f"Stopping early at iteration {i} with cost {costs[i]}")
            rsl = costs[i]
            mu_star = mu_GD[i+1]
            break
    rsl = np.min(costs)
    idx = np.argmin(costs)
    mu_star = mu_GD[idx]
    return rsl, mu_star, value[idx], action[idx], gradient[idx], Rprob_pull_batch[idx], Rprob_donothing_batch[idx]

def main_mp():
    r = 0
    T0=0
    num_proc = 8
    number_of_timeperiods = 15
    number_of_states = 5
    number_of_bandit_in_total = 100
    pulled_ratio = 0.25
    path_length = 10
    num_paths = 5
    sample_number = 100
    (init_prob,
    prob_pull,
    prob_donothing,
    reward,
    radius_pull,
    radius_donothing,
    ), (
    initial_prob,
    transition_kernel_pull,
    transition_kernel_donothing,
    N_sa,
    N_sas,
    )= get_single_sample(
        0,
        number_of_timeperiods,
        number_of_states,
        number_of_bandit_in_total,
        path_length,
        num_paths,
        pulled_ratio,
        r,
    )


    # Robjval_mle_c_b, Rpi1_LP_mle_c_b, Rpi0_LP_mle_c_b = constant_decision_rule_LP(
    #     0,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     initial_prob,
    #     transition_kernel_pull,
    #     transition_kernel_donothing,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     radius_pull,
    #     radius_donothing,
    #     change = 1,
    # )
    # infeasible_Rtotal_reward_mle_c_b, infeasible_active_bandit_number_Rmle_c_b = monte_carlo_randomized_policy_infeasible(
    #     Rpi1_LP_mle_c_b,
    #     Rpi0_LP_mle_c_b,
    #     number_of_timeperiods,
    #     number_of_states,
    #     reward,
    #     prob_pull,
    #     pulled_ratio,
    #     number_of_bandit_in_total,
    #     init_prob,
    # )   

    objval_true_c_b, pi1_LP_true_c_b, pi0_LP_true_c_b = constant_decision_rule_LP(
        0,
        number_of_timeperiods,
        number_of_states,
        reward,
        init_prob,
        prob_pull,
        prob_donothing,
        pulled_ratio,
        number_of_bandit_in_total,
        radius_pull,
        radius_donothing,
        change = 1,
    )
    infeasible_Rtotal_reward_true_c_b, infeasible_active_bandit_number_Rtrue_c_b = monte_carlo_randomized_policy_infeasible(
        pi1_LP_true_c_b,
        pi0_LP_true_c_b,
        number_of_timeperiods,
        number_of_states,
        reward,
        prob_pull,
        pulled_ratio,
        number_of_bandit_in_total,
        init_prob,
    )   

    # value, action, gradient, Rprob_pull, Rprob_donothing, Robust_Lagrangian_relaxation = RLagrangian_relaxation_parallel(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, 0, T0, num_proc)
    Robust_Lagrangian_relaxation_subgradient, mu_star, value, action, gradient, Rprob_pull_batch, Rprob_donothing_batch = subgradient(number_of_timeperiods, number_of_states, reward, init_prob, prob_pull, prob_donothing, pulled_ratio, number_of_bandit_in_total, radius_pull, radius_donothing, T0, num_proc)
    # print(f"Robust_Lagrangian_relaxation", Robust_Lagrangian_relaxation) 
    print(f"Robust_Lagrangian_relaxation_subgradient", Robust_Lagrangian_relaxation_subgradient) 
    print(f"constant_decision_rule_LP", objval_true_c_b) 

class TimeCounter:
    # we implement a simple context timer for measuring the running time of the code
    def __init__(self, name: str = ""):
        self.start_time = time.time()
        self.name = name or "timer"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(
            f"running {self.name} time elapsed: {end_time - self.start_time:.2f} seconds"
        )


if __name__ == "__main__":
    # with TimeCounter("original"):
    #     main()

    with TimeCounter("multiprocessing"):
        main_mp()