import os
import sys
from collections import defaultdict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from utils.model import get_response_log_probs, compute_responsibilities, compute_grad_norm
from utils.config import get_config
from utils.custom import TorchTracemalloc
from utils.data import MBPP_Dataset as CustomDataset
from utils.xformer import load_tokenizer, load_base_model


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
	
	global_step = 0
	
	# ######################################### Initialisation for EM ############################################## #
	# Initialise the model parameters i.e. latent prompt embeddings for each library
	# This is equivalent to latching each library to a random sample from the dataset
	if args.pre_num_iters > 0:
		
		# Set the lr for the initialisation phase
		for param_group in optimizer.param_groups:
			param_group['lr'] = args.init_lr
		
		rdm_idxs = torch.randint(0, len(dataset), (args.num_libraries,))
		if accelerator.is_local_main_process:
			print(f"[DEBUG] Random Indices: {rdm_idxs}")
		
		model.train()
		for k in range(args.num_libraries):
			if accelerator.is_local_main_process:
				logger.info("Initialisation for Library %d", k)
			
			idx = rdm_idxs[k]
			batch = dataset.sample(idx)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			prompt, prompt_mask, response, response_mask = batch
			
			for i in tqdm(range(args.pre_num_iters), desc=f"Init. Iterations Lib {k}", position=0, leave=True):
				
				# Get the response log-probability of the sample coming from the latent prompt of library k
				resp_log_prob = get_response_log_probs(
					args,
					(prompt, prompt_mask, response, response_mask),
					tokenizer,
					model,
					k
				)
				
				loss = -resp_log_prob.sum()  # resp. = 1 for the sample coming from the latent prompt of library k
				
				# Update the model parameters
				accelerator.backward(loss)
				grad_norm = compute_grad_norm(model)
				optimizer.step()
				lr_scheduler.step()  # Make sure this is constant schedule with no warmup
				optimizer.zero_grad()
				
				if args.wandb_logging:
					accelerator.log({
						f'init_loss/clf_{k}': loss.detach().cpu().numpy().item(),
						f'init_grad_norm/clf_{k}': grad_norm
					}, step=global_step)
				
				global_step += 1
			
			with torch.no_grad():
				responsibilities = compute_responsibilities(args, (prompt, prompt_mask, response, response_mask),
															tokenizer, model)
				if accelerator.is_local_main_process:
					logger.info(
						f"[Init Responsibilities, After Training w Cluster {k}] {dataset.ids[idx]}: {responsibilities.cpu().numpy()[0].tolist()}")
					print(
						f"\n[Init Responsibilities, After Training w Cluster {k}] {dataset.ids[idx]}: {responsibilities.cpu().numpy()[0].tolist()}\n")
		
		#  [Debug] Verification - no cluster is empty
		with torch.no_grad():
			for k in range(args.num_libraries):
				idx = rdm_idxs[k]
				batch = dataset.sample(idx)
				batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
				responsibilities = compute_responsibilities(args, batch, tokenizer, model)
				if accelerator.is_local_main_process:
					logger.info(
						f"[Init Responsibilities, Final, GT Cluster {k}] {dataset.ids[idx]}: {responsibilities.cpu().numpy()[0].tolist()}")
					print(
						f"[Init Responsibilities, Final, GT Cluster {k}] {dataset.ids[idx]}: {responsibilities.cpu().numpy()[0].tolist()}")
	
	sys.exit()
	# ############################################################################################################# #
	# ################################################## EM ####################################################### #
	# ############################################################################################################# #
	
	# Wait for all processes to be ready before starting the training loop
	accelerator.wait_for_everyone()
	
	for ep in range(args.num_epochs):
		model.train()
		for _ in tqdm(range(len(train_dataloader)), desc=f"EM Iterations Epoch {ep}", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
							
			# ############################################### E-Step ########################################## #
			# E-Step: Compute resp. corresponding to each program coming from some latent prompt of a library
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
			
			# Posterior probabilities of the sample coming from the latent prompt of each library := p(z_k|x_n)
			responsibilities = compute_responsibilities(args, batch, tokenizer, model)
			
			# ############################################### M-Step ########################################## #
			# M-Step: Update the model parameters i.e. latent prompt embeddings for each library
			#         by maximizing the likelihood of the data coming from the latent prompt of the library
			
			for _ in range(args.max_m_steps):
				q_func = 0  # Total log-likelihood of the data coming from library, metric to track convergence
				lib_train_logs = {}
				
				# ############################# Train clarification by clarification ############################### #
				for k in range(args.num_libraries):
					
					k_responsibilities = responsibilities[:, k]
					# # Re-normalise the respo. for library k
					# k_responsibilities = responsibilities[:, k] / responsibilities[:, k].sum()
					
					# Check k_responsibilities are non-zero
					try:
						assert (k_responsibilities != 0.0).all()
					except AssertionError:
						if accelerator.is_local_main_process:
							logger.info(
								f"Some responsibilities (after norm) for library {k} are still zero = {k_responsibilities}"
							)
					
					# Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
					resp_log_prob = get_response_log_probs(args, batch, tokenizer, model, k)
					
					# Compute Loss = Negative Log Likelihood of the sample coming from the latent prompt of library k
					loss = -(resp_log_prob * k_responsibilities.detach()).sum()
					
					# Update the model parameters
					accelerator.backward(loss)
					grad_norm = compute_grad_norm(model)
					optimizer.step()
					lr_scheduler.step()
					optimizer.zero_grad()
					
					# Update the total log-likelihood of the data coming from library
					q_func += -loss.detach().cpu().numpy().item()
					
					# Bookkeeping
					lib_train_logs[f"loss/clf_{k}"] = loss.detach().cpu().numpy().item()
					lib_train_logs[f"grad_norm/clf_{k}"] = grad_norm
				
				if args.wandb_logging:
					lib_train_logs.update({'q_func': q_func})
					lib_train_logs.update({'lr': lr_scheduler.get_last_lr()[0]})  # Log the learning rate
					accelerator.log(lib_train_logs, step=global_step)
					
				global_step += 1
		

	# ################################################ Evaluate ################################################## #
	# Count the frequency of  the sample-cluster assignments of the trained model
	if accelerator.is_local_main_process:
		logger.info("Starting Evaluation")
	
	model.eval()
	with torch.no_grad():
		posterior_freq_count = defaultdict(int)
		for i in tqdm(range(len(dataset)), desc=f"Evaluating", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
			batch = dataset.sample(i)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			
			# Compute the posterior probabilities
			responsibilities = compute_responsibilities(args, batch, tokenizer, model)
			responsibilities = responsibilities.detach().cpu().numpy()[0]
			
			# Get the library index with the highest responsibility
			library_idx = int(responsibilities.argmax())
			
			# Store the library index
			posterior_freq_count[library_idx] += 1
			
			if accelerator.is_local_main_process:
				logger.info(f"[Posterior] Sample {i} assigned to library {library_idx} from {responsibilities}")
	
	if accelerator.is_local_main_process:
		logger.info("[Final] Posterior Frequency Count: %s", posterior_freq_count)
	
	# ################################################ Save Model ################################################## #
	if accelerator.is_local_main_process:  # only create checkpoint directory on main process
		logger.info("Saving the model at: %s", args.save_at)
		if not os.path.exists(args.save_at):
			os.makedirs(args.save_at)

	accelerator.wait_for_everyone()
	model = accelerator.unwrap_model(model)
	model.save_pretrained(save_directory=args.save_at)  # In place of $ accelerator.save(model.state_dict(), path)


def main():
	args, logger = get_config()
	learn(args, logger)


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file ds_zero3.yaml main_accelerated.py
	main()
	