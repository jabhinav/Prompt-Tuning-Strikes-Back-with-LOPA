import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import torch
from tqdm import tqdm

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftMultiModel
from utils.config import get_config
from utils.model import get_response_embedding, get_response_log_probs_for_lib
from utils.xformer import load_tokenizer, load_base_model

		
def count_num_lib(args, logger):
	with open(os.path.join('./logging/codegen-350m', f'train_most_likely_lib_idx_using_prompt_ll.json'), 'r') as file:
		train_most_likely_lib_idx = json.load(file)
	
	count_dict = defaultdict(int)
	for key in train_most_likely_lib_idx.keys():
		if isinstance(train_most_likely_lib_idx[key], list):
			for idx in train_most_likely_lib_idx[key]:
				count_dict[idx] += 1
		else:
			count_dict[train_most_likely_lib_idx[key]] += 1
	
	# Sort the dict
	count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
	print(count_dict)
	

def get_predicted_clf_freq():
	path = './logging/results_train/mbxp_solutions.json_results.jsonl'
	with open(path, 'r') as file:
		results = file.readlines()
		
	# Get the predictions
	freq_count = defaultdict(int)
	for result in results:
		result = json.loads(result)
		library = result['library']
		library = library.split('_')[-1]
		
		freq_count[library] += 1
		
	# Sort the dict
	freq_count = dict(sorted(freq_count.items(), key=lambda x: x[1], reverse=True))
	print(freq_count)
	

def analyse_passk(args, logger, oracle_results_file):
	"""
	Get the clarification indexes that achieve highest pass@k.
	When done on train set, this will give us the most likely library index for each task to train the prior net.
	"""
	# Read the jsonl file
	data = []
	with open(oracle_results_file, 'r') as file:
		for line in file:
			data.append(json.loads(line))
	
	oracle_results = defaultdict(set)
	for idx in range(0, len(data), args.num_libraries):
		task_id = data[idx]['task_id']
		
		# Create empty set for the task id
		oracle_results[task_id] = set()
		
		# Let's collect the lib idxs which passed
		for i in range(args.num_libraries):
			if data[idx + i]['passed']:
				oracle_results[task_id].add(i)
	
	logger.info("Oracle results")
	oracle_results = {key: list(value) for key, value in oracle_results.items()}
	logger.info(json.dumps(oracle_results, indent=4))
	
	# Create an instance of G.T. from Oracle by randomly choosing one clf idx from passed
	gt = defaultdict(list)
	for key in oracle_results.keys():
		if len(oracle_results[key]) > 0:
			gt[key] = [random.choice(oracle_results[key])]
		else:
			gt[key] = [random.choice(list(range(args.num_libraries)))]
			
	with open(os.path.join(args.log_dir, f'train_gt_instance.json'), 'w') as file:
		json.dump(gt, file)
	
	# Get the problems that are solved by only one clf
	uniques = defaultdict(set)
	for key in oracle_results.keys():
		if len(oracle_results[key]) == 1:
			clf_idx = list(oracle_results[key])[0]
			uniques[clf_idx].add(key)
	
	logger.info("Unique clf idxs")
	uniques = {key: list(value) for key, value in uniques.items()}
	logger.info(json.dumps(uniques, indent=4))
	
	# Get the problems that are solved by only one clf
	logger.info("None solved:")
	none_solved = []
	for key in oracle_results.keys():
		if len(oracle_results[key]) == 0:
			none_solved.append(key)
	logger.info(none_solved)
	logger.info("Number of none solved: {}/{}".format(len(none_solved), len(oracle_results)))
	
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'train_most_likely_lib_idx_using_passk.json'), 'w') as file:
		json.dump(oracle_results, file)


def analyse_multi_passk(args, logger):
	
	# Read each clarification's predictions
	all_predictions = dict()
	for k in range(args.num_libraries):
		path = f'./logging/codegen-350m/PEFT_Oracle_0.50_0.50_20ep/train_split2_results_pass@10/mbxp_solutions_lib_{k}.json_results.jsonl'
		with open(path, 'r') as file:
			results = file.readlines()
		
		all_predictions[k] = dict()
		# Count the number of times clf. passed each task
		for result in results:
			result = json.loads(result)
			task_id = result['task_id']
			
			all_predictions[k][task_id] = all_predictions[k].get(task_id, 0)
			if result['passed']:
				all_predictions[k][task_id] += 1
			
	# Reformate the predictions so that each task has list of number of times a clf idx passed
	hit_distribution = defaultdict(list)
	for k in range(args.num_libraries):
		for task_id in all_predictions[k].keys():
			hit_distribution[task_id].append(all_predictions[k][task_id])
	
	# Now sort the indexes based on the number of times a clf idx passed
	most_likely_solvers = defaultdict(list)
	for task_id in hit_distribution.keys():
		most_likely_solvers[task_id] = sorted(
			[(k, v) for k, v in enumerate(hit_distribution[task_id])],
			key=lambda x: x[1],
			reverse=True
		)

	most_likely_solver = dict()
	for task_id in most_likely_solvers.keys():
		most_likely_solver[task_id] = most_likely_solvers[task_id][0][0]
	
	with open(os.path.join(args.log_dir, f'train_hit_distribution.json'), 'w') as file:
		json.dump(hit_distribution, file)
	with open(os.path.join(args.log_dir, f'train_most_likely_solvers.json'), 'w') as file:
		json.dump(most_likely_solvers, file)
	with open(os.path.join(args.log_dir, f'train_most_likely_lib_idx_using_passk.json'), 'w') as file:
		json.dump(most_likely_solver, file)
	
	# Plot the distribution for each clf idx across all tasks. Each plot must be one below the other
	import matplotlib.pyplot as plt
	for k in range(args.num_libraries):
		plt.figure()
		plt.plot(list(all_predictions[k].keys()), list(all_predictions[k].values()), label=f'clf_{k}')
		plt.title(f'Library {k}')
		plt.xlabel('Tasks')
		plt.ylabel('Number of times the clf idx passed')
		plt.savefig(os.path.join(args.log_dir, f'hist_lib_{k}.png'), dpi=600)
		plt.close()


@torch.no_grad()
def analyse_prompt_ll(args, logger, mode='train'):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
	from utils.data import MBPP_Dataset as CustomDataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode=mode,
		
		# Uncomment to use a finer split of the training data to compute prompt ll
		finer_split=args.finer_train_split,
		use_first_half=args.use_train_first_half
	)
	# [Hack] Manually override the mode to 'test' to get only the question embeddings in the prompt
	dataset.mode = 'test'
	
	# Get the model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# Load checkpoint
	# We load the model state dict on the CPU to avoid an OOM error.
	loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
	loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
	model.load_state_dict(loaded_state_dict, strict=True)
	# release memory
	del loaded_state_dict
	# Log the loaded checkpoint
	message = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
	logger.info(message)
	print(message)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.MULTI_CAUSAL_LM,
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
		num_init_clusters=args.num_libraries,  # My custom field
	)
	# Load the model adapters - in place
	model = PeftMultiModel.from_pretrained(
		model=model,
		model_id=args.load_adapter_from,  # Must be a directory containing the model files
		config=peft_config,
	)
	msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
	logger.info(msg)
	print(msg)
	
	# Update the model's padding token id for open-ended generation
	if 't5' not in args.model_type and model.config.pad_token_id is None:
		model.config.pad_token_id = tokenizer.pad_token_id
	
	# GPU-ize the model
	model.to(args.device)
	most_likely_lib_idx: Dict[str, List[int]] = {}
	for index in tqdm(range(len(dataset)), desc="Getting lib idx", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		prompt, prompt_mask = sample
		
		# Copy the prompt into the response tensor
		response = prompt.clone()
		response[response == tokenizer.pad_token_id] = -100  # Replace pad_token_id with labels [=-100]
		response_mask = response.ne(-100)
		
		sample = (prompt, prompt_mask, response, response_mask)
		
		library_idx = None
		
		max_likelihood = -float('inf')
		for k in range(args.num_libraries):
			with torch.no_grad():
				likelihood = get_response_log_probs_for_lib(args, sample, tokenizer, model, k)
			if likelihood > max_likelihood:
				max_likelihood = likelihood
				library_idx = k
		
		most_likely_lib_idx[dataset.ids[index]] = [int(library_idx)]
	
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'{mode}_most_likely_lib_idx_using_prompt_ll.json'), 'w') as file:
		json.dump(most_likely_lib_idx, file)
		

def use_pred_clf_idxs_to_extract_results(args, pred_clf_idx_file, oracle_results_file, save_with: str = ''):
	
	# Get the predicted clf idxs
	with open(pred_clf_idx_file, 'r') as file:
		test_clf_idxs: Dict[str, List[int]] = json.load(file)
	
	oracle_results = []
	with open(oracle_results_file, 'r') as file:
		for line in file:
			oracle_results.append(json.loads(line))
	
	prior_results = []
	for idx in range(0, len(oracle_results), args.num_libraries):
		task_id = oracle_results[idx]['task_id']
		
		# Store the result for the predicted clf idx
		predicted_clf_idxs = test_clf_idxs[task_id]
		
		if isinstance(predicted_clf_idxs, int):
			predicted_clf_idxs = [predicted_clf_idxs]
		predicted_clf_idx = int(random.choice(predicted_clf_idxs))
		
		prior_results.append(oracle_results[idx + predicted_clf_idx])
	
	# Save the output as jsonl file
	with open(os.path.join(args.log_dir, f'mbxp_solutions_prior_{save_with}.jsonl'), 'w') as file:
		for result in prior_results:
			file.write(json.dumps(result) + '\n')


def main():
	args, logger = get_config()
	
	oracle_results_file = './logging/codegen-350m/PEFT_Oracle_1.0_1.0_5ep/10V/results_train/mbxp_solutions.json_results.jsonl'
	
	# count_num_lib(args, logger)
	
	# # To collect clarification indexes

	# # 1) Using prompt likelihood
	# analyse_prompt_ll(args, logger)
	
	# # 2) Using pass@k
	# analyse_passk(args, logger, oracle_results_file)
	
	# # Use the test clf idxs to compute results
	# for ep in range(-1, 50):
	# 	pred_clf_idx_file = f'./logging/20240111-054218/test_lib_predictions_{ep}.json'
	# 	use_pred_clf_idxs_to_extract_results(args, pred_clf_idx_file, oracle_results_file, save_with=str(ep))
	
	pred_clf_idx_file = f'./logging/max_similarity_clf_idx.json'
	use_pred_clf_idxs_to_extract_results(args, pred_clf_idx_file, oracle_results_file)
	
	# analyse_multi_passk(args, logger)
	

if __name__ == '__main__':
	main()
