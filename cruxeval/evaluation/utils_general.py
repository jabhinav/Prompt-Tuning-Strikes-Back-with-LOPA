# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from utils_execute import check_correctness


def pass_at_k(n, c, k):
    """
    Unbiased estimator of the probability of passing at k.
    :param n: n is the number of generations produced
    :param c: c is the number of correct generations
    :param k: k is the number of generations to consider
    :return:
    """
    if k > n:
        # You need at least k generations to consider. If you have less, you can't pass at k
        return 0.0
    if n - c < k:
        # If c > k correct generations, then every subset of size k will have at least one correct generation
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_score(args):
    gs, (c, i, o), mode, timeout = args

    execution_results = []
    # Loop over all generations for this sample
    for g in gs:
        if mode == "input" and "f(" not in g:  # You want f(input) to be present in the generation
            pass
        elif mode == "output" and f"f({i})" in g:  # You don't want f(input) to be present in the generation
            pass
        else:
            # g := f(input) # Predicated string
            # assert gt_output == f(input) -> This will actually run the code against the input
            # g := output # Predicated string
            # assert gt_output == output -> This will just match the pred. str with the output
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, timeout))
    
    # If no correct generation is found, return False i.e. pass@k = 0 for this sample
    if True not in execution_results:
        execution_results = [False] * len(gs)
    
    return execution_results