import os
import sys
import random
from collections import defaultdict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from utils.config import get_config
from utils.custom import log_dist, is_rank_0
from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.model import get_response_log_probs_for_lib, compute_responsibilities, ClarificationCodeBERTPredictor, \
	compute_grad_norm
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def _process(responsibilities):
	responsibilities = responsibilities.cpu().numpy()[0].tolist()
	# Round off to 2 decimal places each responsibility
	responsibilities = [round(responsibility, 2) for responsibility in responsibilities]
	return responsibilities


@torch.no_grad()
def has_cluster_converged(args, batch, model, tokenizer, k, threshold=0.99):
	model.eval()
	responsibilities = compute_responsibilities(args, batch, tokenizer, model)
	responsibilities = _process(responsibilities)
	return responsibilities[k] >= threshold, responsibilities[k]


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
		if is_rank_0():
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
	
	# Load the library predictor
	prior = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_libraries)
	prior_optimizer = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)
	prior.to(args.device)
	
	model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, train_dataloader, lr_scheduler
	)

	if is_rank_0():
		logger.info("Starting EM for Library Learning")
	
	global_step = 0
	# ######################################### Initialisation for EM ############################################## #
	# Initialise the model parameters i.e. latent prompt embeddings for each library
	# This is equivalent to latching each library to a random sample from the dataset
	
	# Set the lr for the initialisation phase
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.init_lr
	
	# Get random samples from the dataset
	rdm_idxs = random.sample(range(len(dataset)), args.num_libraries)
	
	# To randomly shuffle the order of library initialisation
	rdm_lib_idxs = list(range(args.num_libraries))
	random.shuffle(rdm_lib_idxs)
	
	for z in tqdm(range(args.num_init_epochs), desc=f"Initialisation", position=0, leave=True):

		for k in range(args.num_libraries):
			
			if is_rank_0():
				logger.info(f"({z}) Initialisation for Library {k}")
			
			# Get the sample from the dataset
			idx = rdm_idxs[k]
			batch = dataset.sample(idx)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			
			# Remove the first two tensors from the batch i.e. prior input tensors
			batch = batch[2:]
			
			# Train the model for a few iterations until each cluster has converged
			max_iter = args.pre_num_iters
			curr_iter = 0
			converged, prob = has_cluster_converged(args, batch, model, tokenizer, k)
			pbar = tqdm(desc=f"({z}) Init. Iterations Lib {k}: {prob}", position=0, leave=True, total=max_iter)
			# for i in tqdm(range(args.pre_num_iters), desc=f"Init. Iterations Lib {k}", position=0, leave=True):
			while not converged:
				model.train()
				
				# Get the response log-probability of the sample coming from the latent prompt of library k
				resp_log_prob = get_response_log_probs_for_lib(args, batch, tokenizer, model, k)
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
						f'init_grad_norm/clf_{k}': grad_norm,
						f'init_prob/clf_{k}': prob,
					}, step=global_step)
				
				global_step += 1
				
				pbar.update(1)
				converged, prob = has_cluster_converged(args, batch, model, tokenizer, k)
				pbar.set_description(f"({z}) Init. Iterations Lib {k}: {prob}")
				
				curr_iter += 1
				if curr_iter >= max_iter:
					break
			
			model.eval()
			with torch.no_grad():
				responsibilities = compute_responsibilities(args, batch, tokenizer, model)
				responsibilities = _process(responsibilities)
				if is_rank_0():
					logger.info(f"[({z})Init Responsibilities, After Training w Cluster {k}] {dataset.ids[idx]}: {responsibilities}")
					print(f"\n[({z})Init Responsibilities, After Training w Cluster {k}] {dataset.ids[idx]}: {responsibilities}\n")
		
		#  [Debug] Verification - no cluster is empty
		logger.info("\n\nFinal Init. Responsibilities for :-")
		print("\n\nFinal Init. Responsibilities for :-")
		model.eval()
		with torch.no_grad():
			for k in range(args.num_libraries):
				idx = rdm_idxs[k]
				batch = dataset.sample(idx)
				batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
				
				responsibilities = compute_responsibilities(args, batch, tokenizer, model)
				responsibilities = _process(responsibilities)
				
				if is_rank_0():
					logger.info(f"[({z})Cluster {k}] {dataset.ids[idx]}: {responsibilities}")
					print(f"\n[({z})Cluster {k}] {dataset.ids[idx]}: {responsibilities}\n")
	
	# sys.exit()
	# ############################################################################################################# #
	# ################################################## EM ####################################################### #
	# ############################################################################################################# #
	if is_rank_0():
		logger.info("Starting EM for Clarification Prompt Tuning")
	
	# Reset the lr for the EM phase
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr
	
	# Wait for all processes to be ready before starting the training loop
	accelerator.wait_for_everyone()
	
	for ep in range(args.num_epochs):
		model.train()
		prior.train()
		for _ in tqdm(range(len(train_dataloader)), desc=f"EM Iterations Epoch {ep}", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
							
			# ################################################################################################# #
			# ############################################### E-Step ########################################## #
			# E-Step: Compute resp. corresponding to each program coming from some latent prompt of a library
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
			
			# Posterior probabilities of the sample coming from the latent prompt of each library := p(z_k|x_n)
			responsibilities = compute_responsibilities(args, batch, tokenizer, model, prior)
			
			# ################################################################################################# #
			# ############################################### M-Step ########################################## #
			# M-Step: Update the model parameters i.e. latent prompt embeddings for each clarification
			#         by maximizing the likelihood of the data coming from it
			
			bert_prompt, bert_prompt_mask, prompt, prompt_mask, response, response_mask = batch
			
			# for _ in range(args.max_m_steps):
			q_func = 0  # Total log-likelihood of the data coming from library, metric to track convergence
			lib_train_logs = {}
			
			# ################################### Train the clarification prior ################################ #
			clf_logits = prior(bert_prompt, bert_prompt_mask)
			
			clf_preds = torch.nn.functional.softmax(clf_logits, dim=-1)
			prior_loss = - (responsibilities * torch.log(clf_preds + 1e-8)).sum(dim=-1).mean()
			
			# Compute entropy regularisation
			entropy = - (clf_preds * torch.log(clf_preds + 1e-8)).sum(dim=-1).mean()
			
			# Add entropy regularisation to the loss to avoid local optima
			total_loss = prior_loss - args.ent_coeff * entropy
			
			accelerator.backward(total_loss)
			grad_norm = compute_grad_norm(prior)
			prior_optimizer.step()
			prior_optimizer.zero_grad()
			
			# [Q-function] Initialise with the prior component
			q_func += - prior_loss.detach().cpu().numpy().item()
			
			# Bookkeeping
			lib_train_logs['clf_predictor_loss'] = total_loss.detach().cpu().numpy().item()
			lib_train_logs['clf_predictor_prior_loss'] = prior_loss.detach().cpu().numpy().item()
			lib_train_logs['clf_predictor_entropy'] = entropy.detach().cpu().numpy().item()
			lib_train_logs['grad_norm/prior'] = grad_norm
			
			# ############################# Train clarification by clarification ############################### #
			for k in range(args.num_libraries):
				
				k_responsibilities = responsibilities[:, k]
				# # [For numerical stability] Re-normalise the respo. for library k TODO: affects EM ? (Yes)
				# k_responsibilities = responsibilities[:, k] / responsibilities[:, k].sum()

				# Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
				resp_log_prob = get_response_log_probs_for_lib(
					args,
					(prompt, prompt_mask, response, response_mask),
					tokenizer,
					model,
					k
				)
				# Compute Loss = Negative Log Likelihood of the sample coming from the latent prompt of library k
				loss = -(resp_log_prob * k_responsibilities.detach()).mean()
				
				# Update the model parameters
				accelerator.backward(loss)
				grad_norm = compute_grad_norm(model)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				
				# [Q-function] Add the LL component w.r.t each clf to the Q-function
				q_func += -loss.detach().cpu().numpy().item()
				
				# Bookkeeping
				lib_train_logs[f"loss/clf_{k}"] = loss.detach().cpu().numpy().item()
				lib_train_logs[f"grad_norm/clf_{k}"] = grad_norm
		
			# ######################################## Log the results ###################################### #
			if args.wandb_logging:
				lib_train_logs.update({'q_func': q_func})
				lib_train_logs.update({'lr': lr_scheduler.get_last_lr()[0]})  # Log the learning rate
				accelerator.log(lib_train_logs, step=global_step)
			
			global_step += 1
		
	# ################################################ Evaluate ################################################## #
	# Count the frequency of  the sample-cluster assignments of the trained model
	if is_rank_0():
		logger.info("Starting Evaluation")
	
	model.eval()
	prior.eval()
	with torch.no_grad():
		prior_freq_count = defaultdict(int)
		posterior_freq_count = defaultdict(int)
		for i in tqdm(range(len(dataset)), desc=f"Evaluating", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
			batch = dataset.sample(i)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			
			# Compute the prior probabilities
			prior_logits = prior(batch[0], batch[1])
			prior_probs = torch.nn.functional.softmax(prior_logits, dim=-1)
			prior_probs = prior_probs.detach().cpu().numpy()[0]
			
			# Get the library index with the highest prior probability
			library_idx = int(prior_probs.argmax())
			
			# Store the library index
			prior_freq_count[library_idx] += 1
			
			if is_rank_0():
				logger.info(f"[Prior] Sample {i} assigned to library {library_idx} from  {prior_probs}")
			
			# Compute the posterior probabilities
			responsibilities = compute_responsibilities(args, batch, tokenizer, model, prior)
			responsibilities = responsibilities.detach().cpu().numpy()[0]
			
			# Get the library index with the highest responsibility
			library_idx = int(responsibilities.argmax())
			
			# Store the library index
			posterior_freq_count[library_idx] += 1
			
			if is_rank_0():
				logger.info(f"[Posterior] Sample {i} assigned to library {library_idx} from {responsibilities}")
		
	if is_rank_0():
		logger.info("[Final] Prior Frequency Count: %s", prior_freq_count)
		logger.info("[Final] Posterior Frequency Count: %s", posterior_freq_count)
		
	# ################################################ Save Model ################################################## #
	if is_rank_0():  # only create checkpoint directory on main process
		logger.info("Saving the model at: %s", args.save_at)
		if not os.path.exists(args.save_at):
			os.makedirs(args.save_at)
	
	accelerator.wait_for_everyone()
	model = accelerator.unwrap_model(model)
	model.save_pretrained(save_directory=args.save_at)  # In place of $ accelerator.save(model.state_dict(), path)
	
	prior = accelerator.unwrap_model(prior)
	accelerator.save(prior.state_dict(), os.path.join(args.log_dir, "clf_predictor.pt"))


def main():
	args, logger = get_config()
	
	# Add BERT specific args
	args.bert_model_type = "codebert-base"
	args.bert_tokenizer_name = get_huggingface_path(args.bert_model_type)
	args.bert_model_name_or_path = get_huggingface_path(args.bert_model_type)
	args.bert_config_name = get_huggingface_path(args.bert_model_type)
	msg = f"Added model specific args for {args.bert_model_type}"
	log_dist(message=msg, level=logging.INFO, ranks=[0])
	
	learn(args, logger)


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file ds_zero3.yaml train_bert_coupled_oracle.py
	main()
