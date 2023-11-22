from collections import defaultdict

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from utils.config import get_config
from utils.custom import log_dist
from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.model import get_response_log_probs, compute_responsibilities, ClarificationCodeBERTPredictor, \
	compute_grad_norm
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def learn(args, logger):
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
		sample_problems=args.num_train_problems,
		mode='train'
	)
	
	# args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	args.batch_size = 2
	
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
	
	model = get_peft_model(model, peft_config)
	
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
	
	# ############################################################################################################# #
	# ################################################## EM ####################################################### #
	# ############################################################################################################# #
	logger.info("Starting EM for Clarification Prompt Tuning")
	
	# Let's do EM to update the model with prompt-tuning
	global_step = 0
	
	# Reset the lr for the EM phase
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr
	
	model.train()
	prior.train()
	for ep in range(args.num_epochs):
		for _ in tqdm(range(len(train_dataloader)), desc=f"EM Iterations Epoch {ep}", position=0, leave=True):
			
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
			
			for _ in range(args.max_m_steps):
				lib_train_logs = {}
				
				# ################################### Train the clarification prior ################################ #
				clf_preds = prior(bert_prompt, bert_prompt_mask)
				prior_loss = - (responsibilities * torch.log(clf_preds + 1e-8)).sum(dim=-1).mean()
				# Compute entropy regularisation
				entropy = - (clf_preds * torch.log(clf_preds + 1e-8)).sum(dim=-1).mean()
				# Add entropy regularisation to the loss to avoid local optima
				total_loss = prior_loss - args.ent_coeff * entropy
				
				total_loss.backward()
				grad_norm = compute_grad_norm(prior)
				prior_optimizer.step()
				prior_optimizer.zero_grad()
				
				# [Q-function] Initialise  with the prior component
				q_func = - prior_loss.detach().cpu().numpy().item()
				
				# Bookkeeping
				lib_train_logs['clf_predictor_loss'] = total_loss.detach().cpu().numpy().item()
				lib_train_logs['clf_predictor_prior_loss'] = prior_loss.detach().cpu().numpy().item()
				lib_train_logs['clf_predictor_entropy'] = entropy.detach().cpu().numpy().item()
				lib_train_logs['grad_norm/prior'] = grad_norm
				
				# ############################# Train clarification by clarification ############################### #
				for k in range(args.num_libraries):
					
					k_responsibilities = responsibilities[:, k]
					# # [For numerical stability] Re-normalise the respo. for library k TODO: affects EM (Yes) ?
					# k_responsibilities = responsibilities[:, k] / responsibilities[:, k].sum()
					
					# Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
					resp_log_prob = get_response_log_probs(
						args,
						(prompt, prompt_mask, response, response_mask),
						tokenizer,
						model,
						k
					)
					# Compute Loss = Negative Log Likelihood of the sample coming from the latent prompt of library k
					loss = -(resp_log_prob * k_responsibilities.detach()).mean()
					
					# Update the model parameters
					loss.backward()
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
					# wandb.log(lib_train_logs, step=global_step)
				
				global_step += 1
	
	# ################################################ Evaluate ################################################## #
	# Count the frequency of  the sample-cluster assignments of the trained model
	logger.info("Starting Evaluation")
	
	model.eval()
	prior.eval()
	with torch.no_grad():
		prior_freq_count = defaultdict(int)
		posterior_freq_count = defaultdict(int)
		for i in tqdm(range(len(dataset)), desc=f"Evaluating", position=0, leave=True):
			batch = dataset.sample(i)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
			
			# Compute the prior probabilities
			prior_probs = prior(batch[0], batch[1])
			prior_probs = prior_probs.detach().cpu().numpy()[0]
			
			# Get the library index with the highest prior probability
			library_idx = int(prior_probs.argmax())
			
			# Store the library index
			prior_freq_count[library_idx] += 1
			
			logger.info(f"[Prior] Sample {i} assigned to library {library_idx} from  {prior_probs}")
			
			# Compute the posterior probabilities
			responsibilities = compute_responsibilities(args, batch, tokenizer, model, prior)
			responsibilities = responsibilities.detach().cpu().numpy()[0]
			
			# Get the library index with the highest responsibility
			library_idx = int(responsibilities.argmax())
			
			# Store the library index
			posterior_freq_count[library_idx] += 1
			
			logger.info(f"[Posterior] Sample {i} assigned to library {library_idx} from {responsibilities}")
	
	logger.info("[Final] Prior Frequency Count: %s", prior_freq_count)
	logger.info("[Final] Posterior Frequency Count: %s", posterior_freq_count)


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
