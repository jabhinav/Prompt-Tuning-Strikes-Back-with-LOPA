import json
import os
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftMultiModel
from utils.config import get_config
from utils.custom import TorchTracemalloc, save_predictions_mbxp_format, save_best_lib_predictions_mbxp_format
from utils.data import MBPP_Dataset as CustomDataset
from utils.model import load_tokenizer, load_base_model


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


@torch.no_grad()
def get_library_index(args, prompt, prompt_mask, tokenizer, model):
	"""
	:param args:
	:param prompt: (batch_size, seq_len)
	:param prompt_mask: (batch_size, seq_len)
	:param tokenizer:
	:param model:
	:return: library_idx: int
	"""
	
	batch_size = prompt.size(0)
	max_likelihood = -float('inf')
	library_idx = None
	
	for k in range(args.num_libraries):
		prompt_logits = model(
			library_idx=k,
			input_ids=prompt,
			attention_mask=prompt_mask,
			labels=prompt,
			output_hidden_states=True
		)['logits']
		
		# # Append labels [=-100] for the latent prompt to the prompt
		prefix = torch.full((batch_size, args.num_virtual_tokens), -100).to(prompt.device)
		prompt = torch.cat((prefix, prompt), dim=1)
		# # Append response_mask with 0s for the latent prompt (not the mask for attending to latent prompt)
		prefix_resp_mask = torch.zeros((batch_size, args.num_virtual_tokens)).to(prompt.device)
		prompt_mask = torch.cat((prefix_resp_mask, prompt_mask), dim=1)
		
		prompt[prompt == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
		prompt = prompt.contiguous()
		
		prompt_log_prob = logprobs_from_logits(prompt_logits, prompt)
		prompt_log_prob = prompt_log_prob * prompt_mask
		
		# Likelihood of the prompt coming from the latent prompt of library k
		prompt_log_prob = prompt_log_prob.sum(dim=1)
		
		# Update the library index
		if prompt_log_prob > max_likelihood:
			max_likelihood = prompt_log_prob
			library_idx = k
			
	return library_idx



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
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
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
		msg = (f"trainable params: {trainable_params:,d} || all params: {all_param:,d} ||"
			   f" trainable%: {100 * trainable_params / all_param}")
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
	
	# is_ds_zero_3 = False
	# if getattr(accelerator.state, "deepspeed_plugin", None):
	# 	is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
		
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
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
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
	
	oracle_output: Dict[str, Dict[str, List[str]]] = {}
	lib_mapping:  Dict[str, int] = {}
	
	model.eval()
	
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		input_ids, attention_mask = sample
		
		# ############################## Add logic for identifying the library index ############################## #
		chosen_lib_idx = None
		if args.do_peft:
			chosen_lib_idx = get_library_index(args, input_ids, attention_mask, tokenizer, model)
			
		
		all_library_predictions: Dict[str, List[str]] = OrderedDict()
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
					top_responses = [resp[sample[0].shape[1]:] for resp in top_responses]
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
			
			all_library_predictions[f'lib_{k}'] = all_responses
				
		oracle_output[dataset.ids[index]] = all_library_predictions
		
		if chosen_lib_idx is not None:
			lib_mapping[dataset.ids[index]] = int(chosen_lib_idx)
			
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(oracle_output, f, indent=4)
	
	# Save the output for the best chosen libraries
	if lib_mapping:
		save_best_lib_predictions_mbxp_format(args, oracle_output, lib_mapping, lang='python', d_type='MBPP')
	
	save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')


def main():
	args, logger = get_config()
	# learn(args, logger)
	evaluate(args, logger)


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file ds_zero3.yaml main_accelerated.py
	main()
	