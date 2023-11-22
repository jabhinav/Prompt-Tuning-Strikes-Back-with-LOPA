from collections import defaultdict
from typing import Dict
import json
import os
import torch
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
from typing import Tuple

from utils.model import get_response_embedding, get_response_log_probs, ClarificationCodeBERTPredictor
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path
from utils.config import get_config
from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftMultiModel
from matplotlib import pyplot as plt

	
def train_clf_predictor(args, logger):
	# Add BERT specific args
	args.bert_model_type = "codebert-base"
	args.bert_tokenizer_name = get_huggingface_path(args.bert_model_type)
	args.bert_model_name_or_path = get_huggingface_path(args.bert_model_type)
	args.bert_config_name = get_huggingface_path(args.bert_model_type)
	
	# Load Dataset
	from utils.data import MBPP_Dataset_only_CodeBERT as CustomDataset
	
	tokenizer = load_tokenizer(args.bert_model_type, args.bert_tokenizer_name)
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode='train'
	)
	sampler = RandomSampler(dataset)
	
	args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0, pin_memory=False)
	args.num_training_steps = (len(train_dataloader) * args.num_epochs)
	
	# Get the model
	model = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_libraries)
	
	# Define the optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	
	# GPU-ize the model
	model.to(args.device)
	
	# Train the model
	model.train()
	losses = []
	for ep in range(args.num_epochs):
		epoch_loss = 0

		for j in tqdm(range(len(train_dataloader)), desc=f"EM Iterations Epoch {ep}", position=0, leave=True):
			# Get the batch
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
			
			batch_prompt_embeddings, batch_prompt_mask, batch_lib_idx = batch
			
			# Put the batch on the device
			batch_prompt_embeddings = batch_prompt_embeddings.to(args.device)
			batch_lib_idx = batch_lib_idx.to(args.device)
			
			# Forward pass
			predicted_lib_idx = model(batch_prompt_embeddings, batch_prompt_mask)
			
			# Compute loss
			loss = torch.nn.functional.cross_entropy(predicted_lib_idx, batch_lib_idx)
			
			# Backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.detach().cpu().item()
			losses.append(loss.detach().cpu().item())
			
		# Log the epoch loss
		epoch_loss /= len(train_dataloader)
		logger.info(f"Epoch {ep} loss: {epoch_loss}")
		
		# Save the model
		torch.save(model.state_dict(), os.path.join(args.log_dir, f'lib_predictor.pt'))
		
	# Plot the losses. X-axis is the number of iterations, Y-axis is the loss
	plt.plot(losses)
	plt.savefig(os.path.join(args.log_dir, f'lib_predictor_losses.png'))
	
	# Evaluate the model
	model.eval()
	
	# Sanity check
	train_probs = {}
	train_accuracy = 0
	with torch.no_grad():
		for idx in tqdm(range(len(dataset)), desc="Evaluating", position=0, leave=True):
			sample = dataset.sample(idx)
			sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
			prompt_embedding, prompt_mask, lib_idx = sample
			
			# Get the prediction
			probabilities, indices = model.predict(prompt_embedding, prompt_mask)
			train_probs[dataset.ids[idx]] = probabilities.detach().cpu().numpy()
			
			# if indices.detach().cpu().item() == train_most_likely_lib_idx[dataset.ids[idx]]:
			# 	train_accuracy += 1
		
		# for key in tqdm(train_prompt_embeddings.keys(), desc="Evaluating", position=0, leave=True):
		# 	prompt_embedding = train_prompt_embeddings[key]
		# 	prompt_embedding = prompt_embedding.to(args.device)
		#
		# 	# Get the prediction
		# 	probabilities, indices = model.predict(prompt_embedding)
		# 	train_labels[key] = indices.detach().cpu().item()
		#
		# 	if indices.detach().cpu().item() == train_most_likely_lib_idx[key]:
		# 		train_accuracy += 1
		
	# Save the predictions
	with open(os.path.join(args.log_dir, f'train_lib_predictions.json'), 'w') as file:
		json.dump(train_probs, file, indent=4)
	# print(f"Train accuracy: {train_accuracy / len(train_prompt_embeddings)}")
	
	# test_labels = {}
	# with torch.no_grad():
	# 	for key in tqdm(test_prompt_embeddings.keys(), desc="Evaluating", position=0, leave=True):
	# 		prompt_embedding = test_prompt_embeddings[key]
	# 		prompt_embedding = prompt_embedding.to(args.device)
	#
	# 		# Get the prediction
	# 		probabilities, indices = model.predict(prompt_embedding)
	# 		test_labels[key] = indices.detach().cpu().item()
	#
	# # Save the predictions
	# with open(os.path.join(args.log_dir, f'test_lib_predictions.json'), 'w') as file:
	# 	json.dump(test_labels, file, indent=4)
		

@torch.no_grad()
def save_prompt_embeddings(args, logger, mode):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	from utils.data import MBPP_Dataset as CustomDataset
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode=mode
	)
	# [Hack] Manually override the mode to 'test' to get the question embeddings only in the prompt
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
	
	# Update the model's padding token id for open-ended generation
	if 't5' not in args.model_type and model.config.pad_token_id is None:
		model.config.pad_token_id = tokenizer.pad_token_id
	
	# GPU-ize the model
	model.to(args.device)
	
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
	torch.save(prompt_embeddings, os.path.join(args.log_dir, f'{mode}_prompt_embeddings.pkl'))


@torch.no_grad()
def save_most_likely_lib_idx_using_prompt_ll(args, logger, mode):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
	from utils.data import MBPP_Dataset as CustomDataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode=mode
	)
	# [Hack] Manually override the mode to 'train' to get the question and answer embeddings in the prompt
	dataset.mode = 'train'
	
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
	most_likely_lib_idx: Dict[str, int] = {}
	for index in tqdm(range(len(dataset)), desc="Getting lib idx", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		
		library_idx = None
		
		max_likelihood = -float('inf')
		for k in range(args.num_libraries):
			with torch.no_grad():
				likelihood = get_response_log_probs(args, sample, tokenizer, model, k)
			if likelihood > max_likelihood:
				max_likelihood = likelihood
				library_idx = k
				
		most_likely_lib_idx[dataset.ids[index]] = int(library_idx)
		
		
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'{mode}_most_likely_lib_idx.json'), 'w') as file:
		json.dump(most_likely_lib_idx, file)
		
		
def count_num_lib(args, logger):
	with open(os.path.join('./logging', f'train_most_likely_lib_idx.json'), 'r') as file:
		train_most_likely_lib_idx = json.load(file)
	
	count_dict = defaultdict(int)
	for key in train_most_likely_lib_idx.keys():
		count_dict[train_most_likely_lib_idx[key]] += 1
	
	# Sort the dict
	count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
	print(count_dict)
	

def get_predicted_clf_freq():
	path = './logging/mbxp_solutions_best_lib.json'
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
	

def save_most_likely_lib_idx_using_passk(args, logger, mode):
	# Read the jsonl file
	oracle_results_file = './logging/mbxp_solutions.json_results.jsonl'
	data = []
	with open(oracle_results_file, 'r') as file:
		for line in file:
			data.append(json.loads(line))
			
	oracle_results = dict()
	# Skip to every 5th element
	for idx in range(0, len(data), args.num_libraries):
		task_id = data[idx]['task_id']
		oracle_results[task_id] = []
		
		# Let's collect the lib idxs which passed
		for i in range(args.num_libraries):
			if data[idx + i]['passed']:
				oracle_results[task_id].append(i)
				
		# If none passed, save all the lib idxs
		if len(oracle_results[task_id]) == 0:
			for i in range(args.num_libraries):
				oracle_results[task_id].append(i)
				
				
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'{mode}_most_likely_lib_idx_using_passk.json'), 'w') as file:
		json.dump(oracle_results, file)


def analyse_passk(args, logger):
	# Read the jsonl file
	oracle_results_file = './logging/mbxp_solutions.json_results.jsonl'
	data = []
	with open(oracle_results_file, 'r') as file:
		for line in file:
			data.append(json.loads(line))
	
	oracle_results = defaultdict(set)
	for idx in range(0, len(data), args.num_libraries):
		task_id = data[idx]['task_id']
		
		# Let's collect the lib idxs which passed
		for i in range(args.num_libraries):
			if data[idx + i]['passed']:
				oracle_results[task_id].add(i)
	
	logger.info("Oracle results")
	oracle_results = {key: list(value) for key, value in oracle_results.items()}
	logger.info(json.dumps(oracle_results, indent=4))
	
	uniques = defaultdict(set)
	for key in oracle_results.keys():
		if len(oracle_results[key]) == 1:
			clf_idx = list(oracle_results[key])[0]
			uniques[clf_idx].add(key)
	
	logger.info("Unique clf idxs")
	uniques = {key: list(value) for key, value in uniques.items()}
	logger.info(json.dumps(uniques, indent=4))
	

	
def main():
	args, logger = get_config()
	
	# save_prompt_embeddings(args, logger, mode='train')
	# save_prompt_embeddings(args, logger, mode='test')
	
	# save_most_likely_lib_idx(args, logger, mode='train')
	
	# count_num_lib(args, logger)
	
	# train_clf_predictor(args, logger)
	
	# get_predicted_clf_freq()
	
	# save_most_likely_lib_idx_using_passk(args, logger, mode='train')

	analyse_passk(args, logger)
	

if __name__ == '__main__':
	main()
