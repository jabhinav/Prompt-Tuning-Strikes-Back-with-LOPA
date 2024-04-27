# Copyright (c) Meta Platforms, Inc. and affiliates.

from .base import Task

# import sys
# sys.path.append("..")
from .prompts import (
    make_direct_output_prompt,
    make_direct_output_prompt_phind,
    make_cot_output_prompt,
    make_direct_output_reference
)


class OutputPrediction(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "cruxeval-org/cruxeval"
    DATASET_NAME = None

    def __init__(self, cot = False, phind_output = False):
        self.cot = cot
        self.phind_output = phind_output

        if self.phind_output:
            stop_words = ["# done"]
        else:
            stop_words = ["[/ANSWER]"]

        super().__init__(
            stop_words=stop_words,
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        if self.phind_output:
            return make_direct_output_prompt_phind((doc["code"], doc["input"]))
        elif self.cot:
            return make_cot_output_prompt((doc["code"], doc["input"]))
        else:
            return make_direct_output_prompt((doc["code"], doc["input"]))

    def get_label(self, doc):
        if self.cot:
            print("There is no G.T. label for thought part of the expected response.")
            return None
        else:
            return make_direct_output_reference((doc["code"], doc["input"], doc["output"]))

    def get_reference(self, doc):
        return (doc["code"], doc["input"], doc["output"])

    def postprocess_generation(self, generation, idx):
        # Remove the prompt from the generation
        prompt = self.get_prompt(self.get_dataset()[idx])
        try:
            assert generation.startswith(prompt)
        except AssertionError:
            # Check if there is any non-ascii character in the prompt.
            # Sometimes non-ascii chars in the prompt after encoding and decoding change
            prompt_wo_ascii = prompt.encode('ascii', 'ignore').decode('ascii')
            generation_wo_ascii = generation.encode('ascii', 'ignore').decode('ascii')
            assert generation_wo_ascii.startswith(prompt_wo_ascii)  # This should raise an exception if the prompt is not found in the generation even after removing non-ascii chars
        finally:
            # This will be executed only if the assert statement is not raised inside the except block
            generation = generation[len(prompt):]

        if self.cot:
            # Get the string after the [ANSWER] token
            if "[ANSWER]" in generation:
                generation = generation.split("[ANSWER]")[1].strip()
        
        if "==" in generation:
            # Get the string after the == tokens to get the function output
            generation = generation.split("==")[1].strip()
            
        if "[/ANSWER]" in generation:
            # Get the string before the [/ANSWER] tokens to get the function output
            generation = generation.split("[/ANSWER]")[0].strip()
        
        return generation.strip()

    def process_results(self, generations, references):
        return {}