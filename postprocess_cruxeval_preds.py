import json
import os
import utils.cruxeval_tasks as cruxeval_tasks
import argparse

"""
The idea:

Many times, some models are predicting a space before the last token of the prompt which is '\n'. Maybe tokenizer is
decoding it in that way. So we remove the '\n' from the end of the prompt and then remove the prefix of the prediction.
Note: This is a very specific post-processing step for the CruxEval tasks and specific models.
"""


def find_mismatch(_prompt, _gen):
	# Iterate through the characters of both strings
	for i in range(len(_prompt)):
		if _prompt[i] != _gen[i]:
			# If a mismatch is found, return the index of the mismatched character in y
			return i
	# If no mismatch is found, check if x is a prefix of y
	if len(_prompt) <= len(_gen):
		return -1  # x is a prefix of y
	else:
		return len(_gen)  # y is shorter than x, so mismatch occurs at y's end


def post_process(args):
	task = cruxeval_tasks.get_task(f'{args.mode}_prediction', cot=False)
	dataset = task.get_dataset()
	dataset_rows = range(dataset.num_rows)
	dataset = dataset.add_column("row_index", dataset_rows)
	dataset = dataset.filter(lambda example: example['row_index'] >= len(dataset) // 2)
	id_to_idx = {}
	for i in range(len(dataset)):
		sample = dataset[i]
		id_to_idx[sample["id"]] = sample["row_index"]
	
	assert os.path.exists(args.path), f"Path does not exist: {args.path}"
	raw_output = json.load(open(args.path, 'r'))
	
	processed_output = {}
	for sample_id, preds in raw_output.items():
		processed_output[sample_id] = []
		for pred in preds:
			
			# Process here
			_idx = id_to_idx[sample_id]
			prompt = task.get_prompt(task.get_dataset()[_idx])
			
			if len(pred) <= len(prompt):
				print("Found prediction with length less than or equal to prompt length. Skipping it: ", sample_id)
				processed_output[sample_id].append('')
				continue
			
			# Find the mismatching index
			if not pred.startswith(prompt):
				
				mismatch_at = find_mismatch(prompt, pred)
				if mismatch_at != -1:
					prompt = prompt[:mismatch_at]
					assert pred.startswith(prompt), f"prompt: {prompt}, pred: {pred}"
					processed_pred = pred[len(prompt):]
					processed_pred = task.postprocess_generation(pred, _idx, gen_with_prompt_removed=processed_pred)
				
				else:
					processed_pred = task.postprocess_generation(pred, _idx)
			
			else:
				processed_pred = task.postprocess_generation(pred, _idx)
				
			processed_output[sample_id].append(processed_pred)
			
			
	# Save the processed output
	new_path = args.path.replace('output_raw', 'output')
	print("Saving the processed output to:", new_path)
	with open(new_path, 'w') as f:
		json.dump(processed_output, f, indent=4)
		

if __name__ == '__main__':
	# Read the predictions file
	# mode = 'output'
	# model = 'Meta-Llama-3-8B'
	# path = f'logging/cruxeval_{mode}/{model}_raw_results/output_raw.json'
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default=None)
	parser.add_argument('--mode', type=str, default=None)
	_args = parser.parse_args()
	
	post_process(_args)
