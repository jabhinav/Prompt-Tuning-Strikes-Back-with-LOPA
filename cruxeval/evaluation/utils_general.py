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
    gs, (c, i, o), mode = args

    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            pass
        elif mode == "output" and f"f({i})" in g:
            pass
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results