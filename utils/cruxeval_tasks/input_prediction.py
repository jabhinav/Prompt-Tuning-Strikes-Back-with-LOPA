# Copyright (c) Meta Platforms, Inc. and affiliates.

from .base import Task

# import sys
# sys.path.append("..")
from .prompts import (
    make_direct_input_prompt,
    make_cot_input_prompt,
    make_direct_input_reference
)


class InputPrediction(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "cruxeval-org/cruxeval"
    DATASET_NAME = None

    def __init__(self, cot = False):
        self.cot = cot
        super().__init__(
            stop_words=["[/ANSWER]"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        if self.cot:
            return make_cot_input_prompt((doc["code"], doc["output"]))
        else:
            return make_direct_input_prompt((doc["code"], doc["output"]))
        
    def get_label(self, doc):
        if self.cot:
            print("There is no G.T. label for thought part of the expected response.")
            return None
        else:
            return make_direct_input_reference((doc["code"], doc["input"], doc["output"]))

    def get_reference(self, doc):
        return (doc["code"], doc["input"], doc["output"])

    def postprocess_generation(self, generation, idx, gen_with_prompt_removed=None):
        
        if gen_with_prompt_removed is None:
            # Remove the prompt from the generation
            prompt = self.get_prompt(self.get_dataset()[idx])
            try:
                assert generation.startswith(prompt)
            except AssertionError:
                # Check if there is any non-ascii character in the prompt.
                # Sometimes non-ascii chars in the prompt after encoding and decoding change
                prompt_wo_ascii = prompt.encode('ascii', 'ignore').decode('ascii')
                generation_wo_ascii = generation.encode('ascii', 'ignore').decode('ascii')
                assert generation_wo_ascii.startswith(
                    prompt_wo_ascii)  # This should raise an exception if the prompt is not found in the generation even after removing non-ascii chars
            finally:
                # This will be executed only if the assert statement is not raised inside the except block
                generation = generation[len(prompt):]
        else:
            generation = gen_with_prompt_removed
        
        if self.cot:
            # Get the string after the [ANSWER] token
            if "[ANSWER]" in generation:
                generation = generation.split("[ANSWER]")[1].strip()
        
        # Get the string before the == tokens to get the function input
        if "==" in generation:
            generation = generation.split("==")[0].strip()
        
        # Remove the `assert f` from the generation
        if "assert f" in generation:
            generation = "f" + generation.split("assert f")[1].strip()
        
        return generation.strip()

    def process_results(self, generations, references):
        return {}