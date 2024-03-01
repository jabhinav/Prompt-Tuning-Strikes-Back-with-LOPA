import os
import sys

import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftMultiModel, get_peft_model
from utils.config import get_config
from utils.custom import log_dist, is_rank_0, unwrap_model
from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.model import compute_grad_norm, ClarificationCodeBERTPredictor, get_response_log_probs_for_cVAE, \
	gumbel_softmax
from utils.xformer import load_tokenizer, get_huggingface_path, load_base_model


class cVAE(torch.nn.Module):
	def __init__(self, args, tokenizer):
		super().__init__()
		
		self.args = args
		self.tokenizer = tokenizer
		
		# Encoder
		self.enc = self.init_encoder()
		self.dec = self.init_decoder()
	
	def init_encoder(self):
		"""
		Initialize the encoder.
		"""
		args = self.args
		model = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_virtual_tokens)
		
		if args.clf_predictor_path is not None:
			# Load the model state dict on the CPU to avoid an OOM error.
			loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
			loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
			model.load_state_dict(loaded_state_dict, strict=True)
			
			# release memory
			del loaded_state_dict
			
			# Log the loaded checkpoint
			msg = "Loaded encoder checkpoint from path: {}".format(args.clf_predictor_path)
			if is_rank_0():
				print(msg)
		
		return model
	
	def init_decoder(self):
		"""
		Initialize the decoder.
		:return:
		"""
		
		args = self.args
		
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
			msg = "Loaded decoder base checkpoint from path: {}".format(args.load_base_from_path)
			if is_rank_0():
				print(msg)
		
		# Get the config
		peft_config = PromptTuningConfig(
			task_type=TaskType.cVAE_CAUSAL_LM,
			# CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
			# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
			# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
		)
		
		if args.load_adapter_from is not None:
			# Load the model adapters - in place
			model = PeftMultiModel.from_pretrained(
				model=model,
				model_id=args.load_adapter_from,  # Must be a directory containing the model files
				config=peft_config,
			)
			msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
			if is_rank_0():
				print(msg)
		else:
			# Initialize the model adapters
			model = get_peft_model(model, peft_config)
		
		return model
	
	def get_nb_trainable_parameters(self):
		
		enc_trainable_params = sum(p.numel() for p in self.enc.parameters() if p.requires_grad)
		enc_all_params = sum(p.numel() for p in self.enc.parameters())
		dec_trainable_params, dec_all_params = self.dec.get_nb_trainable_parameters()
		
		return enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params
	
	def encode(self, input_ids, attention_mask):
		clf_logits = self.enc(input_ids, attention_mask=attention_mask)
		return clf_logits
	
	def decode(self, batch, latent_attention_weights):
		resp_log_prob = get_response_log_probs_for_cVAE(
			self.args,
			batch,
			self.tokenizer,
			self.dec,
			latent_attention_weights
		)
		return resp_log_prob
	

	
	def forward(self, batch):
		
		# Encode
		clf_logits = self.enc(*batch[:2])
		
		# Sample using Gumbel Softmax
		k_vector = gumbel_softmax(clf_logits)
		
		# Decode: Multiply k_vector by the log probability of the response for corresponding k
		resp_log_prob = self.decode(batch[2:], k_vector)
		
		# Loss
		reconstruction_loss = -resp_log_prob.sum(dim=-1).mean()
		
		# KL Divergence
		kl_div = torch.sum(k_vector * torch.log(k_vector * self.args.num_virtual_tokens + 1e-8), dim=-1).mean()
		
		# Total loss
		loss = reconstruction_loss + kl_div
		
		return loss, reconstruction_loss, kl_div


def _save(args, model: cVAE, force_dir=None):
	if force_dir is not None and not os.path.exists(force_dir):
		os.makedirs(force_dir)
	
	# Save Encoder
	if force_dir is not None:
		torch.save(model.enc.state_dict(), os.path.join(force_dir, "clf_predictor.pt"))
	else:
		torch.save(model.enc.state_dict(), os.path.join(args.log_dir, "clf_predictor.pt"))
	
	# Save Decoder
	if force_dir is not None:
		model.dec.save_pretrained(save_directory=force_dir)
	else:
		model.dec.save_pretrained(
			save_directory=args.save_at)  # In place of $ accelerator.save(model.state_dict(), path)
	
	if is_rank_0():
		if force_dir is not None:
			print("Saved the Decoder model at:", force_dir)
			print("Saved the Encoder model at:", os.path.join(force_dir, "clf_predictor.pt"))
		else:
			print("Saved the Decoder model at:", args.save_at)
			print("Saved the Encoder model at:", os.path.join(args.log_dir, "clf_predictor.pt"))


def learn(args, logger):
	experiment_config = vars(args)
	if args.wandb_logging:
		wandb.init(project=args.project_name, config=experiment_config)
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
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
	model = cVAE(args, tokenizer)
	model.to(args.device)
	
	# Print the number of trainable parameters
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
	
	# Get the learning rate scheduler
	lr_scheduler = get_constant_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=0,
	)
	
	# ############################################################################################################# #
	# ################################################## VAE ###################################################### #
	# ############################################################################################################# #
	global_step = 0
	
	# ############################################################################################################ #
	# ##################################################### Train ################################################ #
	for ep in range(args.num_epochs):
		model.train()
		for _ in tqdm(range(len(train_dataloader)), desc=f"VAE Iterations Epoch {ep}", position=0, leave=True):
			
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
			
			# Compute Loss = Forward pass through cVAE
			loss, reconstruction_loss, kl_div = model(batch)
			
			# Update the model parameters
			loss.backward()
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
			lib_train_logs.update({'lr': lr_scheduler.get_last_lr()[0]})  # Log the learning rate
			if args.wandb_logging:
				wandb.log(lib_train_logs, step=global_step)
			
			global_step += 1
			
			break  # Break after 1 iteration for debugging
	
	# ############################################################################################################ #
	# ################################################ Evaluate ################################################## #
	# # Count the frequency of  the sample-cluster assignments of the trained model
	# if is_rank_0():
	# 	logger.info("Starting Evaluation")
	#
	# model.eval()
	# with torch.no_grad():
	# 	for i in tqdm(range(len(dataset)), desc=f"Evaluating", position=0, leave=True,
	# 				  disable=not accelerator.is_local_main_process):
	# 		batch = dataset.sample(i)
	# 		batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
	#
	# 		# Compute the prior probabilities
	# 		clf_logits = model.enc(batch[0], batch[1])
	# 		clf_probs = torch.nn.functional.softmax(clf_logits, dim=-1)
	# 		clf_probs = clf_probs.detach().cpu().numpy()[0]
	#
	# 		if is_rank_0():
	# 			logger.info(f"[Prior] Sample {i} latent attention weights: {clf_probs}")
	
	# ################################################ Save Model ################################################## #
	if is_rank_0():  # only create checkpoint directory on main process
		logger.info("Saving the model at: %s", args.save_at)
		if not os.path.exists(args.save_at):
			os.makedirs(args.save_at)
	
	_save(args, unwrap_model(model))


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
