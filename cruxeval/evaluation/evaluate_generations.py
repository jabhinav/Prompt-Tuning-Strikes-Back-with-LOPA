# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils_general import (
    evaluate_score,
    pass_at_k,
)


def evaluate_generations(generations, mode, timeout=3):
    """
    I changed the logic from evaluating all generations for every sample in the dataset to evaluating just the samples
    for which generations are provided. This should be useful if you want to evaluate on a subset of the dataset.
    :param generations:
    :param mode:
    :param timeout:
    :return:
    """
    samples_considered = list(generations.keys())
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/cruxeval.jsonl")
    dataset = [json.loads(l) for l in open(path_to_data, "r").readlines()]
    references = [
        (doc["code"], doc["input"], doc["output"])
        for doc in dataset if doc["id"] in samples_considered
    ]

    # Run the samples
    try:
        generations_list = [generations[doc["id"]] for doc in dataset if doc["id"] in samples_considered]
    except:
        assert False, "check format of generations, should be dictionary of lists with keys of id's in the form sample_i"
        
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations_list, references, [mode] * len(generations_list), [timeout] * len(generations_list))
        _results = executor.map(evaluate_score, args_list)
    
    all_scores = list(_results)

    # Compute pass@k scores
    pass_at_1s, pass_at_5s = [], []
    for execution_result in all_scores:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
        pass_at_5s.append(pass_at_k(n, c, 5))

    return {
        "raw_generations": generations,
        "raw_scored_generations": {
            doc_id: all_scores[i] for i, doc_id in enumerate(samples_considered)
        },
        "pass_at_1": sum(pass_at_1s) / len(pass_at_1s) * 100,
        "pass_at_5": sum(pass_at_5s) / len(pass_at_5s) * 100
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_path", 
        help="JSON path containing outputs to evaluate. Should contain a list of \
              length 800, where each element is a list of different generations \
              for that benchmark sample.",
        type=str,
    )
    parser.add_argument(
        "--scored_results_path", 
        help="path to dump scored results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mode", 
        help="either input or output, depending on which one to evaluate",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--timeout",
        help="timeout for each execution (in seconds)",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    generations = json.load(open(args.generations_path))
    # print(f"Scoring {args.generations_path}... expect around a minute")

    # # No need for this, since the mode is provided as an argument -> Following can give wrong results if
    # if "input" in args.generations_path:
    #     args.mode = "input"
    # else:
    #     args.mode = "output"

    results = evaluate_generations(generations, args.mode, args.timeout)
    
    # print(f"Finished!")
    # print("pass@1:", round(results["pass_at_1"], 1), "pass@5:", round(results["pass_at_5"], 1))
    print(round(results["pass_at_1"], 1))
    if args.scored_results_path is not None:
        # print(f"Dumping to {args.scored_results_path}")
        json.dump(results, open(args.scored_results_path, "w"))
