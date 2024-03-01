import json
import os
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftMultiModel
from utils.config import get_config
from utils.data import MBPP_Dataset as CustomDataset
from utils.model import get_response_embedding, get_clf_embedding
from utils.xformer import load_tokenizer, load_base_model


@torch.no_grad()
def get_prompt_embeddings(args, logger):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
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
	msg = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
	logger.info(msg)
	print(msg)
		
	model.to(args.device)
	
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode=args.data_modality,
		
		# Uncomment to use a finer split of the training data to evaluate
		finer_split=args.finer_train_split,
		use_first_half=args.use_train_first_half
	)
	
	# [Hack] Manually override the mode to 'test' to get the question embeddings only in the prompt
	dataset.mode = 'test'
	
	# Iterate through the dataset sample-by-sample
	model.eval()
	prompt_embeddings: Dict[str, torch.Tensor] = {}
	for index in tqdm(range(len(dataset)), desc="Getting prompt embedding", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		prompt, prompt_mask = sample
		
		# Copy the prompt into the response tensor
		response = prompt.clone()
		response[response == tokenizer.pad_token_id] = -100  # Replace pad_token_id with labels [=-100]
		response_mask = response.ne(-100)
		
		# Get the prompt embedding
		prompt_embedding = get_response_embedding(model, prompt, prompt_mask, response)
		
		prompt_embeddings[dataset.ids[index]] = prompt_embedding.detach().cpu()
	
	
	# Save the output as pkl file
	torch.save(prompt_embeddings, os.path.join(args.log_dir, args.prompt_embedding_fname))


@torch.no_grad()
def get_clf_embeddings(args, logger):
	
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
	msg = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
	logger.info(msg)
	print(msg)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.MULTI_CAUSAL_LM,  # CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
		# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
		# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
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
	model.to(args.device)
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode=args.data_modality,
		
		# Uncomment to use a finer split of the training data to evaluate
		finer_split=args.finer_train_split,
		use_first_half=args.use_train_first_half
	)
	
	# [Hack] Manually override the mode to 'test' to get the question embeddings only in the prompt
	dataset.mode = 'test'
	
	# Iterate through the dataset sample-by-sample
	model.eval()
	clf_embeddings: Dict[int, torch.Tensor] = {}
	
	def get_sample(idx):
		sample = dataset.sample(idx)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		prompt, prompt_mask = sample
		
		# Copy the prompt into the response tensor
		response = prompt.clone()
		response[response == tokenizer.pad_token_id] = -100  # Replace pad_token_id with labels [=-100]
		
		return prompt, prompt_mask, response
	
	
	prompt, prompt_mask, response = get_sample(0)
	
	for k in tqdm(range(args.num_libraries), desc="Getting Clf embedding", position=0, leave=True):

		# Get the clf embedding
		clf_embedding = get_clf_embedding(args, model, prompt, prompt_mask, response, k)
		
		clf_embeddings[k] = clf_embedding.detach().cpu()
	
	# Save the output as pkl file
	torch.save(clf_embeddings, os.path.join(args.log_dir, args.clf_embedding_fname))


def compute_similarity(prompt_embedding, clf_embedding):
	similarity = torch.cosine_similarity(prompt_embedding, clf_embedding, dim=-1)
	return similarity


def get_max_sim_clf_idxs(args, logger):
	
	print("Identifying the clf index with the highest similarity to the prompt embedding")
	logger.info("Identifying the clf index with the highest similarity to the prompt embedding")
	
	# Load the prompt and clf embeddings
	prompt_embeddings = torch.load(os.path.join(args.log_dir, args.prompt_embedding_fname))
	clf_embeddings = torch.load(os.path.join(args.log_dir, args.clf_embedding_fname))
	
	similarity_file = {}
	for task_id in prompt_embeddings.keys():
		prompt_embedding = prompt_embeddings[task_id]
		similarities = {}
		for clf_idx, clf_embedding in clf_embeddings.items():
			similarity = compute_similarity(prompt_embedding, clf_embedding)
			similarities[clf_idx] = similarity.detach().cpu().numpy().item()
		similarity_file[task_id] = similarities
	
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'similarity_file.json'), 'w') as f:
		json.dump(similarity_file, f, indent=4)
		
	max_similarity_clf_idx_file = {}
	for task_id, similarities in similarity_file.items():
		# Get the clf index with the highest similarity
		max_similarity_clf_idx_file[task_id] = max(similarities, key=similarities.get)
	
	
	# Analyse -> Compute frequency of clf idxs
	clf_idx_freq = {}
	for task_id, clf_idx in max_similarity_clf_idx_file.items():
		if clf_idx in clf_idx_freq:
			clf_idx_freq[clf_idx] += 1
		else:
			clf_idx_freq[clf_idx] = 1
	logger.info(f"Frequency of clf idxs: {clf_idx_freq}")
	print(f"Frequency of clf idxs: {clf_idx_freq}")
	
	# Save the output as json file
	with open(os.path.join(args.log_dir, args.pred_clf_idx_fname), 'w') as f:
		json.dump(max_similarity_clf_idx_file, f, indent=4)


def use_pred_clf_idxs_to_extract_results(args, save_with: str = ''):
	# Get the predicted clf idxs
	with open(os.path.join(args.log_dir, args.pred_clf_idx_fname), 'r') as file:
		test_clf_idxs: Dict[str, List[int]] = json.load(file)
	
	oracle_results = []
	with open(args.oracle_results_file, 'r') as file:
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
	
	# Override relevant arguments
	args.load_base_from_path = './logging/codegen-350m/Baseline_0.75/output/pytorch_model.bin'
	args.load_adapter_from = './logging/codegen-350m/PEFT_Oracle_0.75_0.25_5ep/PromptTuningMultiModel'
	args.oracle_results_file = './logging/codegen-350m/PEFT_Oracle_0.75_0.25_5ep/results/mbxp_solutions.json_results.jsonl'
	
	args.prompt_embedding_fname = 'prompt_embeddings.pkl'
	args.clf_embedding_fname = 'clf_embeddings.pkl'
	args.pred_clf_idx_fname = 'max_similarity_clf_idx.json'
	
	args.data_modality = 'test'
	args.finer_train_split = 0.50
	args.use_train_first_half = False
	
	get_prompt_embeddings(args, logger)
	get_clf_embeddings(args, logger)
	get_max_sim_clf_idxs(args, logger)
	use_pred_clf_idxs_to_extract_results(args, save_with='emb_similarity')
	
	
if __name__ == "__main__":
	main()