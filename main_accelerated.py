import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftMultiModel
from utils.custom import TorchTracemalloc, create_dir, is_rank_0, log_dist, set_dist, set_seed, \
	save_predictions_mbxp_format
from utils.data import MBPP_Dataset as CustomDataset
from utils.model import load_tokenizer, load_base_model, get_huggingface_path


def logprobs_from_logits(logits, labels):
	"""
	See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
	"""
	log_p = F.log_softmax(logits, dim=2)
	logpy = torch.gather(log_p, 2, labels.unsqueeze(2)).squeeze(-1)
	return logpy


def get_response_log_probs(args, batch, tokenizer, model, library_idx):
	prompt, prompt_mask, response, response_mask = batch
	batch_size = prompt.size(0)
	
	# # Set the library index
	# print("[Debug] Library Index:", library_idx)
	# model.library_idx = library_idx
	# print("[Debug] Model Library Index:", model.library_idx)
	
	resp_logits = model(
		library_idx=library_idx,
		input_ids=prompt,
		attention_mask=prompt_mask,
		labels=response,
		output_hidden_states=True
	)['logits']
	
	# # Append labels [=-100] for the latent prompt to the response
	prefix = torch.full((batch_size, args.num_virtual_tokens), -100).to(response.device)
	response = torch.cat((prefix, response), dim=1)
	# # Append response_mask with 0s for the latent prompt (not the mask for attending to latent prompt)
	prefix_resp_mask = torch.zeros((batch_size, args.num_virtual_tokens)).to(response_mask.device)
	response_mask = torch.cat((prefix_resp_mask, response_mask), dim=1)
	
	response[response == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
	resp_labels = response.contiguous()
	
	resp_log_prob = logprobs_from_logits(resp_logits, resp_labels)
	resp_log_prob = resp_log_prob * response_mask
	
	# Likelihood of the sample coming from the latent prompt of library k
	resp_log_prob = resp_log_prob.sum(dim=1)
	
	return resp_log_prob


@torch.no_grad()
def compute_responsibilities(args, batch, tokenizer, model) -> torch.Tensor:
	"""
	Compute the responsibilities i.e. posterior probabilities of the sample coming from the latent prompt of each
	library.
	:param args:
	:param batch:
	:param tokenizer:
	:param model:
	:return:
	"""
	
	batch_size = batch[0].size(0)
	
	# Create a tensor of shape (n_samples, num_libraries) to store the responsibilities
	likelihood = torch.zeros((batch_size, args.num_libraries)).to(args.device)
	
	for k in range(args.num_libraries):
		# Store the likelihood of the sample coming from the latent prompt of library k
		likelihood[:, k] = get_response_log_probs(args, batch, tokenizer, model, k)
	
	# Normalize the responsibilities (prior can be uniform, thus cancelled out)
	responsibilities = F.softmax(likelihood, dim=1)
	
	return responsibilities.detach()


def learn(args, logger):
	# No need to specify fp16 while launching script with $ accelerate launch
	# (since we specify it during $ accelerate config) else specify mixed_precision = args.mixed_precision
	if args.wandb_logging:
		accelerator = Accelerator(log_with=["wandb"])
	else:
		accelerator = Accelerator()
	
	# We need to initialize the trackers we use, and also store our configuration
	experiment_config = vars(args)
	accelerator.init_trackers(args.project_name, experiment_config)
	args.device = accelerator.device  # Overwrite device to use accelerator device
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.MULTI_CAUSAL_LM,  # CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
		# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
		# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
		num_init_clusters=args.num_libraries,  # My custom field
	)
	
	# Get the dataset
	with accelerator.main_process_first():
		dataset = CustomDataset(
			path_to_data=args.path_to_data,
			tokenizer=tokenizer,
			max_prompt_length=args.max_prompt_length,
			max_length=args.max_length,
			sample_problems=args.num_train_problems,
			mode='train'
		)
	
	args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	
	# Prepare training data loader
	sampler = RandomSampler(dataset)
	train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0, pin_memory=False)
	args.num_training_steps = (len(train_dataloader) * args.num_epochs)
	
	# Get the model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path
	)
	# Load checkpoint
	if args.load_base_from_path is not None:
		# We load the model state dict on the CPU to avoid an OOM error.
		loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		msg = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		if accelerator.is_local_main_process:
			logger.info(msg)
			print(msg)
	
	model = get_peft_model(model, peft_config)
	
	if accelerator.is_local_main_process:
		trainable_params, all_param = model.get_nb_trainable_parameters()
		msg = f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
		logger.info(msg)
		print(msg)  # Prompt tuning: embedding_dim * num_virtual_tokens * num_libraries
	
	# Get the optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	lr_scheduler = get_constant_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
	)
	
	model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, train_dataloader, lr_scheduler
	)
	
	is_ds_zero_3 = False
	if getattr(accelerator.state, "deepspeed_plugin", None):
		is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
		
	if accelerator.is_local_main_process:
		logger.info("Starting EM for Library Learning")
	
	# # Debug Load the model
	# model.load_adapter(model_id=args.save_at, adapter_name='default')
	
	# ######################################### Initialisation for EM ############################################## #
	# Initialise the model parameters i.e. latent prompt embeddings for each library
	# This is equivalent to latching each library to a random sample from the dataset
	if args.pre_num_iters > 0:
		rdm_idxs = torch.randint(0, len(dataset), (args.num_libraries,))
		for k in range(args.num_libraries):
			with TorchTracemalloc() as tracemalloc:
				model.train()
				
				if accelerator.is_local_main_process:
					logger.info("Initialisation for Library %d", k)
				
				for i in tqdm(range(args.pre_num_iters), desc=f"Init. Iterations Lib {k}", position=0, leave=True):
					batch = dataset.sample(rdm_idxs[k])
					batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
					
					# Get the response log-probability of the sample coming from the latent prompt of library k
					resp_log_prob = get_response_log_probs(args, batch, tokenizer, model, k)
					
					loss = -resp_log_prob.sum()  # resp. = 1 for the sample coming from the latent prompt of library k
					
					# Update the model parameters
					accelerator.backward(loss)
					optimizer.step()
					lr_scheduler.step()  # Make sure this is constant schedule with no warmup
					optimizer.zero_grad()
					
					if accelerator.is_local_main_process:
						logger.info(f"Iter {i} Loss: {loss.detach().cpu().numpy().item()}")
	
	# ################################################## EM ####################################################### #
	# Let's do EM to update the model with prompt-tuning
	global_step = 0
	for _ in tqdm(range(args.num_epochs), desc="Epochs", position=0, leave=True):
		for _ in tqdm(range(len(train_dataloader)), desc="EM Iterations", position=0, leave=True):
			
			with TorchTracemalloc() as tracemalloc:
				model.train()
				
				# ############################################### E-Step ########################################## #
				# E-Step: Compute resp. corresponding to each program coming from some latent prompt of a library
				batch = next(iter(train_dataloader))
				batch = tuple(t.to(args.device) for t in batch)
				
				# Posterior probabilities of the sample coming from the latent prompt of each library := p(z_k|x_n)
				responsibilities = compute_responsibilities(args, batch, tokenizer, model)
				
				# To prevent underflow, clip the responsibilities to a minimum value
				responsibilities = responsibilities.clamp(min=1e-8)
				
				# ############################################### M-Step ########################################## #
				# M-Step: Update the model parameters i.e. latent prompt embeddings for each library
				#         by maximizing the likelihood of the data coming from the latent prompt of the library
				
				q_func = 0  # Total log-likelihood of the data coming from library, metric to track convergence
				responsibilities.to(args.device)
				
				# Library Book-keeping
				lib_train_logs = {}
				for k in range(args.num_libraries):
					
					# Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
					resp_log_prob = get_response_log_probs(args, batch, tokenizer, model, k)
					
					# Re-normalise the respo. for library k -> Avoids numerical instability and does not affect EM
					norm_responsibilities = responsibilities[:, k] / responsibilities[:, k].sum()
					
					# Check norm_responsibilities are non-zero
					try:
						assert (norm_responsibilities != 0).all()
					except AssertionError:
						if accelerator.is_local_main_process:
							logger.info(
								f"Some responsibilities (after norm) for library {k} are still zero = {norm_responsibilities}"
							)
					
					# Compute Loss = Negative Log Likelihood of the sample coming from the latent prompt of library k
					loss = -(resp_log_prob * norm_responsibilities.detach()).sum()
					
					# Update the model parameters
					accelerator.backward(loss)
					optimizer.step()
					lr_scheduler.step()
					optimizer.zero_grad()
					
					# Update the total log-likelihood of the data coming from library
					q_func += -loss.detach().cpu().numpy().item()
					
					# Bookkeeping
					lib_train_logs[f"loss/lib_{k}"] = loss.detach().cpu().numpy().item()
				
				if accelerator.is_local_main_process:
					logger.info("Iteration: %d, Q-Func: %.4f", global_step, q_func)
				
				if args.wandb_logging:
					lib_train_logs.update({'q_func': q_func})
					accelerator.log(lib_train_logs, step=global_step)
					
				global_step += 1
	
	# ################################################ Save Model ################################################## #
	if accelerator.is_local_main_process:  # only create checkpoint directory on main process
		logger.info("Saving the model at: %s", args.save_at)
		if not os.path.exists(args.save_at):
			os.makedirs(args.save_at)

	accelerator.wait_for_everyone()
	model = accelerator.unwrap_model(model)
	model.save_pretrained(save_directory=args.save_at)  # In place of $ accelerator.save(model.state_dict(), path)
	
	# ####################################### Compute final responsibilities ####################################### #
	if args.infer_final_responsibilities:
		if accelerator.is_local_main_process:
			logger.info("\n\n# ################# Computing Final Responsibilities ################# #")
		
		model.eval()
		for i in tqdm(range(len(dataset)), desc="Computing Final Responsibilities", position=0, leave=True):
			batch = dataset.sample(i)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			
			responsibilities = compute_responsibilities(args, batch, tokenizer, model)
			if accelerator.is_local_main_process:
				# Debug by showing the responsibilities of each sample
				logger.info(f"[Responsibilities] {dataset.ids[i]}: {responsibilities.cpu().numpy()[0].tolist()}")


@torch.no_grad()
def evaluate(args, logger):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.MULTI_CAUSAL_LM,  # CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
		# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
		# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
		num_init_clusters=args.num_libraries,  # My custom field
	)
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test'
	)
	
	# Get the model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path
	)
	
	# Load checkpoint
	if args.load_base_from_path is not None:
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
	
	if args.do_peft:
		
		if args.load_adapter_from is None:
			logger.error("Please specify the path to load the model adapters from")
			raise ValueError("Please specify the path to load the model adapters from")
		
		# Load the model adapters - in place
		model = PeftMultiModel.from_pretrained(
			model=model,
			model_id=args.load_adapter_from,  # Must be a directory containing the model files
			config=peft_config,
		)
		logger.info("Loaded the model adapters from: %s", args.load_adapter_from)
	
	# GPU-ize the model
	model.to(args.device)
	
	# Predict for each sample output by each library
	num_loops = int(args.num_return_sequences / args.num_return_sequences_per_iter)
	
	output: Dict[str, Dict[str, List[str]]] = {}
	model.eval()
	
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		batch = dataset.sample(index)
		batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
		input_ids, attention_mask = batch
		
		library_predictions: Dict[str, List[str]] = OrderedDict()
		for k in range(args.num_libraries):
			
			# Set the library index
			if args.do_peft:
				model.library_idx = k
			
			all_responses: List[str] = []
			try:
				
				for _ in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
					top_responses = model.generate(
						input_ids=input_ids,
						attention_mask=attention_mask,
						max_new_tokens=args.max_new_tokens,
						do_sample=args.do_sample,
						num_beams=args.num_beams,
						early_stopping=True if args.num_beams > 1 and not args.do_sample else False,
						temperature=args.temperature if args.do_sample else 1.0,
						top_p=args.top_p if args.do_sample else 1.0,
						num_return_sequences=args.num_return_sequences_per_iter,
					)
				
					top_responses = top_responses.detach().cpu().numpy().tolist()
					top_responses = [resp[batch[0].shape[1]:] for resp in top_responses]
					top_responses = [tokenizer.decode(resp, skip_special_tokens=False) for resp in top_responses]
					# Split the response at the first occurrence of the end of text token.
					# This works since we append the eos token to responses and make the model predict it
					# Also useful to not consider any text after the first occurrence of the eos token
					top_responses = [resp.split(tokenizer.eos_token)[0] for resp in top_responses]
					all_responses.extend(top_responses)
					
			except Exception as e:
				if isinstance(e, UnboundLocalError) and str(
						e) == "local variable 'next_tokens' referenced before assignment":
					# See https://github.com/huggingface/transformers/issues/5118
					logger.exception("Problem text was > specified tokens, so cannot do generation")
					logger.info(e)
					raise e
				else:
					logger.exception("Unexpected exception in generating solution")
					logger.info(e)
					raise e
				
				# # Default to empty string on errors
				# prediction = ""

			# # For APPS
			# if len(prediction):
			# 	prediction = prediction.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
			
			library_predictions[f'lib_{k}'] = all_responses
				
		output[dataset.ids[index]] = library_predictions
			
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(output, f, indent=4)
	
	save_predictions_mbxp_format(args, logger, output, lang='python', d_type='MBPP')
	

def get_config():
	# Create a directory to store the logs
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join('./logging', current_time)
	create_dir(log_dir)
	
	# Configure logging
	if is_rank_0():
		logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
							format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger(__name__)
	logger.info("\n\n# ################# Learning Libraries ################# #\n\n")
	
	# Define the parameters
	model_type = "codegen2-1B"  # codegen2-1B, codegen-350M, CodeLlama-7b-Python-hf
	huggingface_path = get_huggingface_path(model_type)
	
	parser = argparse.ArgumentParser()
	
	# High-level
	parser.add_argument('--wandb_logging', type=bool, default=False)
	parser.add_argument('--project_name', type=str, default='PromptTuningModel')
	parser.add_argument('--do_peft', type=bool, default=False)
	parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
	
	# Paths
	parser.add_argument('--path_to_data', type=str, default='./data/MBPP/mbpp_release_v1.jsonl')
	parser.add_argument('--save_at', type=str, default=log_dir + '/PromptTuningMultiModel')
	parser.add_argument('--load_adapter_from', type=str, default=None)  # Path to dir
	parser.add_argument('--load_base_from_path', type=str, default=None)
	
	# Prompt Tuning Parameters
	parser.add_argument('--max_prefix_length', type=int, default=0)
	parser.add_argument('--num_virtual_tokens', type=int, default=10)
	parser.add_argument('--max_prompt_length', type=int, default=325)  # Max 384
	parser.add_argument('--max_length', type=int, default=325+256)  # Max 384+512
	parser.add_argument('--num_libraries', type=int, default=5)
	
	# Model
	parser.add_argument("--model_type", type=str, default=model_type)
	parser.add_argument("--model_name_or_path", type=str, default=huggingface_path)
	parser.add_argument("--config_name", type=str, default=huggingface_path)
	parser.add_argument("--tokenizer_name", type=str, default=huggingface_path)
	
	# Training
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--pre_num_iters", type=int, default=0)
	parser.add_argument("--per_gpu_train_batch_size", type=int, default=1)
	parser.add_argument("--lr", type=float, default=5e-5)
	
	# Others
	parser.add_argument("--num_test_problems", type=int, default=None, choices=[None, 100])
	parser.add_argument("--num_train_problems", type=int, default=None, choices=[None, 100])
	parser.add_argument("--infer_final_responsibilities", type=bool, default=False)
	
	# Evaluation
	parser.add_argument("--save_results_at", type=str, default=os.path.join(log_dir, 'all_codes.json'))
	parser.add_argument("--num_beams", type=int, default=1)
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--do_sample", type=bool, default=True)
	parser.add_argument("--num_return_sequences", type=int, default=5)
	parser.add_argument("--num_return_sequences_per_iter", type=int, default=5)
	parser.add_argument("--temperature", type=float, default=0.6)
	parser.add_argument("--top_p", type=float, default=0.95)
	
	# Hardware configuration
	parser.add_argument("--no_cuda",
						help="Avoid using CUDA when available")
	parser.add_argument('--fp16', default=True, action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training (multi-node): local_rank")
	parser.add_argument('--db', default=False,
						help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')
	parser.add_argument("--node_index", type=int, default=-1,
						help="node index if multi-node running")
	parser.add_argument("--gpu_per_node", type=int, default=4,
						help="num of gpus per node")
	
	args = parser.parse_args()
	
	args.log_dir = log_dir
	# Update the max_length and max_prompt_length by deducting the number of virtual tokens
	args.max_length = args.max_length - args.num_virtual_tokens
	args.max_prompt_length = args.max_prompt_length - args.num_virtual_tokens
	# Update the number of lib if peft is not used
	if not args.do_peft:
		args.num_libraries = 1
	
	set_dist(args, logger)
	set_seed(args)
	
	# Log the config
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	log_dist(logger, message="\n\n# ############### PEFT ############## #\n\n", level=logging.INFO, ranks=[0])
	log_dist(logger, message=json.dumps(config, indent=4), level=logging.INFO, ranks=[0])
	
	return args, logger


def main():
	args, logger = get_config()
	# learn(args, logger)
	evaluate(args, logger)


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file ds_zero3.yaml main_accelerated.py
	main()
	