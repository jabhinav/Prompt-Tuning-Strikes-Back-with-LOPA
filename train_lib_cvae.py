import os
from collections import defaultdict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from utils.config import get_config
from utils.custom import log_dist, is_rank_0
from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.model import Lib_cVAE, compute_grad_norm
from utils.xformer import load_tokenizer, get_huggingface_path


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
	model = Lib_cVAE(args, tokenizer)
	
	if accelerator.is_local_main_process:
		enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params = model.get_nb_trainable_parameters()
		msg = (f"Encoder: trainable params: {enc_trainable_params:,d} || all params: {enc_all_params:,d} ||"
			   f" trainable%: {100 * enc_trainable_params / enc_all_params}")
		logger.info(msg)
		print(msg)
		
		msg = (f"Decoder: trainable params: {dec_trainable_params:,d} || all params: {dec_all_params:,d} ||"
			   f" trainable%: {100 * dec_trainable_params / dec_all_params}")
		logger.info(msg)
		print(msg)
	
	# Get the optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	lr_scheduler = get_constant_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
	)
	
	model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, train_dataloader, lr_scheduler
	)

	# ############################################################################################################# #
	# ################################################## VAE ###################################################### #
	# ############################################################################################################# #
	global_step = 0
	
	# Wait for all processes to be ready before starting the training loop
	accelerator.wait_for_everyone()
	
	# ############################################################################################################ #
	# ##################################################### Train ################################################ #
	for ep in range(args.num_epochs):
		model.train()
		for _ in tqdm(range(len(train_dataloader)), desc=f"VAE Iterations Epoch {ep}", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
			
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
	
			# Compute Loss = Forward pass through cVAE
			loss, reconstruction_loss, kl_div = model(batch)
			
			# Update the model parameters
			accelerator.backward(loss)
			grad_norm = compute_grad_norm(model)
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()
			
			# Bookkeeping
			lib_train_logs = {
				f"loss": loss.detach().cpu().numpy().item(),
				f"reconstruction_loss": reconstruction_loss.detach().cpu().numpy().item(),
				f"kl_div": kl_div.detach().cpu().numpy().item(),
				f"grad_norm": grad_norm
			}
			
			# ######################################## Log the results ###################################### #
			if args.wandb_logging:
				lib_train_logs.update({'lr': lr_scheduler.get_last_lr()[0]})  # Log the learning rate
				accelerator.log(lib_train_logs, step=global_step)
			
			global_step += 1
			
	############################################################################################################ #
	################################################ Evaluate ################################################## #
	# Count the frequency of  the sample-cluster assignments of the trained model
	if is_rank_0():
		logger.info("Starting Evaluation")

	model.eval()
	with torch.no_grad():
		prior_freq_count = defaultdict(int)
		for i in tqdm(range(len(dataset)), desc=f"Evaluating", position=0, leave=True,
					  disable=not accelerator.is_local_main_process):
			batch = dataset.sample(i)
			batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)

			# Compute the prior probabilities
			prior_logits = model.enc(batch[0], batch[1])
			prior_probs = torch.nn.functional.softmax(prior_logits, dim=-1)
			prior_probs = prior_probs.detach().cpu().numpy()[0]

			# Get the library index with the highest prior probability
			library_idx = int(prior_probs.argmax())

			# Store the library index
			prior_freq_count[library_idx] += 1

			if is_rank_0():
				logger.info(f"[Prior] Sample {i} assigned to library {library_idx} from  {prior_probs}")

	if is_rank_0():
		logger.info("[Final] Prior Frequency Count: %s", prior_freq_count)
	
	# ################################################ Save Model ################################################## #
	if is_rank_0():  # only create checkpoint directory on main process
		logger.info("Saving the model at: %s", args.save_at)
		if not os.path.exists(args.save_at):
			os.makedirs(args.save_at)
	
	model._save(accelerator)


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
	# To run with accelerate, $ accelerate launch --config_file ds_zero3_cpu_nofp16.yaml train_cvae.py --load_base_from_path ./logging/codegen-350m/Baseline_1.0/output/pytorch_model.bin --do_peft 1
	main()
