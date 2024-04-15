"""
In this version of train_cvae, encoder and decoder are not wrapped in a dictionary.

Benefits:
- This gives freedom to choose different optimizers (with different lr) and schedulers for the encoder and decoder.

Drawbacks:
- Can't use deepspeed with accelerator since it requires a single model.
"""


import math
import os
import time

import accelerate
import torch
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import MultiProcessAdapter
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

import logging
from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftCvaeModel
from utils.config import get_config
from utils.custom import log_dist, is_rank_0
from utils.data import MBPP_Dataset_wEnc as CustomDataset
from utils.model import PromptSpecificClarificationEncoder as EmbeddingEncoder
from utils.xformer import load_base_model
from utils.xformer import load_tokenizer, get_huggingface_path


class Trainer(object):
	
	def __init__(self, args, logger):
		self.args = args
		
		# init with accelerate
		self._init_accelerator()
		self.accelerator.wait_for_everyone()
		
		with self.accelerator.main_process_first():
			self.logger = logger
		
		# Log some info
		self.logger.info("=" * 56)
		self.logger.info("||\t\t" + "New training process started." + "\t\t||")
		self.logger.info("=" * 56)
		self.logger.info("\n")
		self.logger.info(f"Experiment name: {args.project_name}")
		self.logger.info(f"Experiment directory: {self.args.log_dir}")
		
		# init counts
		self.batch_count: int = 0
		self.step: int = 0
		self.epoch: int = 0
		
		# Init temperature
		self.curr_temp = 1.0
		
		# setup tokenizer
		logger.info(f"Loading Dec tokenizer from {get_huggingface_path(args.model_type)}")
		self.tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
		logger.info(f"Loading Enc tokenizer from {get_huggingface_path(args.enc_model_type)}")
		self.enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))
		
		# setup data_loader
		with self.accelerator.main_process_first():
			self.logger.info("Building dataset...")
			start = time.monotonic_ns()
			self.train_dataset, self.train_dataloader = self._build_dataloader()
			end = time.monotonic_ns()
			self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")
		
		# setup model
		with self.accelerator.main_process_first():
			self.logger.info("Building model...")
			start = time.monotonic_ns()
			self.model = self._build_model()
			end = time.monotonic_ns()
			self.logger.debug(self.model)
			self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
			# Print the number of trainable parameters
			enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params = self.__count_parameters(self.model)
			msg = (f"Encoder: trainable params: {enc_trainable_params:,d} || all params: {enc_all_params:,d} ||"
				   f" trainable%: {100 * enc_trainable_params / enc_all_params}")
			self.logger.info(msg)
			msg = (f"Decoder: trainable params: {dec_trainable_params:,d} || all params: {dec_all_params:,d} ||"
				   f" trainable%: {100 * dec_trainable_params / dec_all_params}")
			self.logger.info(msg)
		
		# optimizer & scheduler
		with self.accelerator.main_process_first():
			self.logger.info("Building optimizer and scheduler...")
			start = time.monotonic_ns()
			self.optimizer = self._build_optimizer()
			self.scheduler = self._build_scheduler()
			end = time.monotonic_ns()
			self.logger.info(
				f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
			)
		
		# accelerate prepare
		self.logger.info("Initializing accelerate...")
		start = time.monotonic_ns()
		self._accelerator_prepare()
		end = time.monotonic_ns()
		self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")
		
		# We need to recalculate our total training steps as the size of the training dataloader may have changed after
		# Accelerator's prepare function.
		self.recalculate_training_metrics()
		
		# save config file path
		self.config_save_path = os.path.join(self.args.log_dir, "args.json")
		self.device = self.accelerator.device
	
	def _init_accelerator(self):
		
		project_config = ProjectConfiguration(
			logging_dir=self.args.log_dir,
		)
		kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		
		if self.args.wandb_logging:
			self.accelerator = accelerate.Accelerator(
				gradient_accumulation_steps=self.args.gradient_accumulation_steps,
				log_with=["wandb"],
				project_config=project_config,
				kwargs_handlers=[kwargs],
			)
		else:
			self.accelerator = accelerate.Accelerator(
				gradient_accumulation_steps=self.args.gradient_accumulation_steps,
				project_config=project_config,
				kwargs_handlers=[kwargs],
			)
		
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": "cVAE"}}
			)
	
	def _build_dataloader(self):
		# Get the dataset
		dataset = CustomDataset(
			path_to_data=self.args.path_to_data,
			tokenizer=self.tokenizer,
			max_prompt_length=self.args.max_prompt_length,
			max_length=self.args.max_length,
			sample_problems=self.args.num_train_problems,
			mode='train',
			enc_tokenizer=self.enc_tokenizer,
		)
		
		# Prepare training data loader
		sampler = RandomSampler(dataset)
		train_dataloader = DataLoader(
			dataset,
			sampler=sampler,
			batch_size=self.args.per_gpu_train_batch_size,
			num_workers=0,
			pin_memory=False
		)
		
		num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.accelerator.gradient_accumulation_steps)
		self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
		
		return dataset, train_dataloader
	
	def init_encoder(self):
		"""
		Initialize the encoder.
		"""
		args = self.args
		model = EmbeddingEncoder(
			args=self.args,
			n_virtual_tokens=self.args.total_virtual_tokens,
			word_embedding_dim=self.args.word_embedding_dim
		)
		
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
			task_type=TaskType.CVAE_CAUSAL_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
		)
		
		if args.load_adapter_from is not None:
			# Load the model adapters - in place
			model = PeftCvaeModel.from_pretrained(
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
			
		# This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
		self.args.total_virtual_tokens = self.args.num_virtual_tokens * peft_config.num_transformer_submodules
		self.args.word_embedding_dim = peft_config.token_dim
		
		return model
	
	def _build_model(self):
		
		model = {
			"dec": self.init_decoder(),  # First init the decoder to set eff. num virtual tokens and word embedding dim
			"enc": self.init_encoder(),
		}
		
		return model
	
	def _build_optimizer(self):
		optimizer_e = torch.optim.AdamW(self.model['enc'].parameters(), lr=self.args.enc_lr)
		optimizer_d = torch.optim.AdamW(self.model['dec'].parameters(), lr=self.args.dec_lr)
		return {"optimizer_enc": optimizer_e, "optimizer_dec": optimizer_d}
		
	def _build_scheduler(self):
		# Get the learning rate scheduler
		scheduler_e = get_constant_schedule_with_warmup(
			optimizer=self.optimizer['optimizer_enc'],
			num_warmup_steps=0,
		)
		scheduler_d = get_constant_schedule_with_warmup(
			optimizer=self.optimizer['optimizer_dec'],
			num_warmup_steps=0,
		)
		
		return {"scheduler_enc": scheduler_e, "scheduler_dec": scheduler_d}
	
	
		
	
	def _ds_prepare(self):
		# [TO AVOID] You must specify a training or evaluation dataloader in accelerate.prepare() when using DeepSpeed
		# Debug: https://github.com/huggingface/accelerate/pull/676
		AcceleratorState().deepspeed_plugin.deepspeed_config[
			"train_micro_batch_size_per_gpu"] = self.args.per_gpu_train_batch_size
	
	def _accelerator_prepare(self):
		"""
		Can't use this script with deepspeed since it requires preparation of a single model.
		Issue: https://github.com/huggingface/accelerate/issues/253
		:return:
		"""
		
		self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
		
		if isinstance(self.model, dict):
			for key in self.model.keys():
				self.model[key] = self.accelerator.prepare(self.model[key])
		else:
			self.model = self.accelerator.prepare(self.model)
		
		if isinstance(self.optimizer, dict):
			for key in self.optimizer.keys():
				self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
		else:
			self.optimizer = self.accelerator.prepare(self.optimizer)
		
		if isinstance(self.scheduler, dict):
			for key in self.scheduler.keys():
				self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
		else:
			self.scheduler = self.accelerator.prepare(self.scheduler)
	
	
	def recalculate_training_metrics(self):
		
		num_update_steps_per_epoch = math.ceil(
			len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps)
		self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
		
		# # After wards we recalculate our number of training epochs.
		# Keep this. Useful when max_train_steps is to be set manually
		self.args.num_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
		self.args.total_batch_size = (
				self.args.per_gpu_train_batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
		)
		
		self.logger.info("\n")
		self.logger.info(f"  Num examples = {len(self.train_dataset)}")
		self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
		self.logger.info(f"  Instantaneous batch size per device = {self.args.per_gpu_train_batch_size}")
		self.logger.info(
			f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.total_batch_size}")
		self.logger.info(f"  Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}")
		self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
		self.logger.info("\n")
		
	
	@staticmethod
	def __count_parameters(model):
		enc_trainable_params = sum(p.numel() for p in model['enc'].parameters() if p.requires_grad)
		enc_all_params = sum(p.numel() for p in model['enc'].parameters())
		dec_trainable_params, dec_all_params = model['dec'].get_nb_trainable_parameters()
		return enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params
	
	
	@staticmethod
	def gumbel_softmax(logits, temperature=1.0, eps=1e-20):
		"""
		See: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
		"""
		U = torch.rand_like(logits)
		gumbel = -torch.log(-torch.log(U + eps) + eps)
		y = logits + gumbel
		return F.softmax(y / temperature, dim=-1)
	
	
	def set_gumbel_temp(self, min_temp: float = 0.5, decay_rate: float = 2e-3, update_every: int = 50):
		
		if self.step % update_every == 0:
			# Decay the temperature according to max(min_temp, exp(-decay_rate * step))
			curr_temp = max(min_temp, math.exp(-decay_rate * self.step))
			self.curr_temp = curr_temp
	
	
	@staticmethod
	def logprobs_from_logits(logits, labels):
		"""
		See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
		"""
		log_p = F.log_softmax(logits, dim=2)
		logpy = torch.gather(log_p, 2, labels.unsqueeze(2)).squeeze(-1)
		return logpy
	
	
	def get_response_log_probs(self, batch, tokenizer, model, latent_attention_weights):
		"""
		:param batch:
		:param tokenizer:
		:param model:
		:param latent_attention_weights:
		:return: Log-likelihood of the response per sample
		"""
		prompt, prompt_mask, response, response_mask = batch
		batch_size = prompt.size(0)
		
		# # Set the library index
		# print("[Debug] Library Index:", library_idx)
		# model.library_idx = library_idx
		# print("[Debug] Model Library Index:", model.library_idx)
		
		output = model(
			latent_attention_weights=latent_attention_weights,
			input_ids=prompt,
			attention_mask=prompt_mask,
			labels=response,
			output_hidden_states=True
		)
		resp_logits = output.logits
		
		# # Prepare the response mask for the complete response (including for the latent prompt)
		# Append response_mask with 0s for the latent prompt (this is not the mask for attending to latent prompt)
		latent_response_mask = torch.zeros((batch_size, self.args.num_virtual_tokens)).to(response_mask.device)
		response_mask = torch.cat((latent_response_mask, response_mask), dim=1)
		
		# # Prepare labels for the complete response (including for the latent prompt)
		# Append labels [=-100] for the latent prompt to the response
		latent_response = torch.full((batch_size, self.args.num_virtual_tokens), -100).to(response.device)
		response = torch.cat((latent_response, response), dim=1)
		response[response == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
		
		# # Prepare the response labels and mask for computing the log-probability
		shift_response = response[..., 1:].contiguous()
		shift_logits = resp_logits[..., :-1, :].contiguous()
		shift_mask = response_mask[..., 1:].contiguous().float()
		
		# # Compute the log-probability of the response tokens
		resp_log_prob = self.logprobs_from_logits(shift_logits, shift_response)
		resp_log_prob *= shift_mask  # Mask out the padding tokens
		
		# # Average the log-probability across the tokens. This will return the log-likelihood of the response per sample
		# # For normalised
		resp_log_prob = resp_log_prob.sum(dim=1)/shift_mask.sum(dim=1)
		# # [Uncomment] For un-normalised
		# resp_log_prob = resp_log_prob.sum(dim=1)
		
		# NOTE: (-)Mean of resp_log_prob across samples won't match the XE loss returned as output.loss because
		# the XE loss flattens the log probabilities of the labels across the batch
		
		# # [For Debugging Only] see if the negative log-probability matches the XE loss returned as output.loss
		# # Torch XE loss flattens the log probabilities of the labels across the batch
		# resp_log_prob = self.logprobs_from_logits(shift_logits, shift_response)
		# resp_log_prob = resp_log_prob.view(-1)
		# shift_mask = shift_mask.view(-1)
		# resp_log_prob *= shift_mask
		# # Compute the negative log-probability
		# nll = -resp_log_prob.sum()/shift_mask.sum()
		
		return resp_log_prob
	
	
	def get_reconstruction_loss(self, batch, model, latent_attention_weights):
		prompt, prompt_mask, response, response_mask = batch
		
		# # Set the library index
		# print("[Debug] Library Index:", library_idx)
		# model.library_idx = library_idx
		# print("[Debug] Model Library Index:", model.library_idx)
		
		output = model(
			latent_attention_weights=latent_attention_weights,
			input_ids=prompt,
			attention_mask=prompt_mask,
			labels=response,
			output_hidden_states=True
		)
		
		return output.loss
	
	
	def _train_step(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		# Encode
		att_logits = self.model["enc"](*batch[:2])
		
		# # [Uncomment to] Set att_weights as sigmoid of att_logits
		att_weights = torch.sigmoid(att_logits)
		
		# Loss function = -log(p(x|z))
		reconstruction_loss = self.get_reconstruction_loss(
			batch[2:],
			self.model["dec"],
			latent_attention_weights=att_weights
		)
		
		# # [Uncomment to] use KL Divergence = KL(q(z|x) || p(z))
		# kl_div = torch.sum(att_weights * torch.log(att_weights * self.args.num_virtual_tokens + 1e-8), dim=-1).mean()
		
		# Total loss
		# total_loss = reconstruction_loss + self.args.kl_coeff * kl_div
		total_loss = reconstruction_loss
		
		# BP and Grad Updated
		self.optimizer["optimizer_enc"].zero_grad()
		self.optimizer["optimizer_dec"].zero_grad()
		self.accelerator.backward(total_loss)
		self.optimizer["optimizer_enc"].step()
		self.optimizer["optimizer_dec"].step()
		
		return {
			f"total_loss": total_loss.detach().cpu().numpy().item(),
			f"reconstruction_loss": reconstruction_loss.detach().cpu().numpy().item(),
			# f"kl_div": kl_div.detach().cpu().numpy().item(),
		}, {
			f"gumbel_temp": self.curr_temp,
		}
	
	def _train_epoch(self):
		r"""Training epoch. Should return average loss of a batch (sample) over
		        one epoch. See ``train_loop`` for usage.
		"""
		
		# Set the models to train mode
		self.model["enc"].train()
		self.model["dec"].train()
		
		epoch_losses: dict = {}
		epoch_step: int = 0
		for batch in tqdm(
				self.train_dataloader,
				desc=f"Training Epoch {self.epoch}",
				unit="batch",
				colour="GREEN",
				leave=False,
				dynamic_ncols=True,
				smoothing=0.04,
				disable=not self.accelerator.is_main_process,
		):
			# with self.accelerator.accumulate(self.model):
			train_losses, train_stats = self._train_step(batch)
			
			self.batch_count += 1
			
			if self.batch_count % self.args.gradient_accumulation_steps == 0:

				for key, value in train_losses.items():
					if key not in epoch_losses.keys():
						epoch_losses[key] = value
					else:
						epoch_losses[key] += value
				
				if self.args.wandb_logging:
					self.accelerator.log(
						{
							"Step/Total Loss": train_losses["total_loss"],
							"Step/Reconstruction Loss": train_losses["reconstruction_loss"],
							# "Step/KL Divergence": train_losses["kl_div"],
							"Step/Gumbel Temperature": train_stats["gumbel_temp"],
							"Step/Encoder Learning Rate": self.optimizer[
								"optimizer_enc"
							].param_groups[0]["lr"],
							"Step/Decoder Learning Rate": self.optimizer[
								"optimizer_dec"
							].param_groups[0]["lr"],
						},
						step=self.step,
					)
				self.step += 1
				epoch_step += 1
		
		self.accelerator.wait_for_everyone()
		
		# Compute the average losses for the epoch
		for key in epoch_losses.keys():
			epoch_losses[key] = (
					epoch_losses[key] / len(self.train_dataloader) * self.args.gradient_accumulation_steps
			)
		
		return epoch_losses

	@torch.no_grad()
	def eval_epoch(self):
		
		# Set the models to eval mode
		self.model["enc"].eval()
		self.model["dec"].eval()
		
		# Create a file to save the attention weights
		with open(os.path.join(self.args.log_dir, f"attention_weights_{self.epoch}.txt"), "w") as f:
			
			# We only need to save the attention weights for each training sample
			batch_count = 0
			for batch in tqdm(
					self.train_dataloader,
					desc=f"Evaluating Epoch {self.epoch}",
					unit="batch",
					colour="GREEN",
					leave=False,
					dynamic_ncols=True,
					smoothing=0.04,
					disable=not self.accelerator.is_main_process,
			):
				f.write("-" * 32 + "\n")
				f.write("Batch {}: \n".format(batch_count))
				
				# Encode
				clf_logits = self.model["enc"](*batch[:2])
				
				# # [Alternate] Set att_weights as sigmoid of att_logits
				att_weights = torch.sigmoid(clf_logits)
				
				# Log the attention weights. Round to 4 decimal places
				att_weights = torch.round(att_weights * 1e4) / 1e4
				f.write("[Predicted] Attention weights:\n {}\n".format(att_weights.detach().cpu().numpy()))
				
				batch_count += 1

	
	def train_loop(self):
		r"""Training loop. The public entry of training process."""

		self.accelerator.wait_for_everyone()
		while self.epoch < self.args.num_epochs:
			self.logger.info("\n")
			self.logger.info("-" * 32)
			self.logger.info("Epoch {}: ".format(self.epoch))
			
			# Do training epoch
			train_losses = self._train_epoch()
			
			if isinstance(train_losses, dict):
				for key, loss in train_losses.items():
					self.logger.info("  |- Train/{}: {:.6f}".format(key, loss))
					
					if self.args.wandb_logging:
						self.accelerator.log(
							{"Epoch/{}".format(key): loss},
							step=self.step,
						)
						
			# Do evaluation epoch
			self.eval_epoch()

			# Update info for each epoch
			self.epoch += 1
			
			if self.epoch % self.args.save_every == 0:
				self.accelerator.wait_for_everyone()
				if self.accelerator.is_main_process:
					self.save(f"epoch_{self.epoch}")
		
		# Finish training and save final checkpoint
		self.accelerator.wait_for_everyone()
		if self.accelerator.is_main_process:
			# self.accelerator.save_state(os.path.join(self.args.log_dir, "final_epoch"))
			self.save("final")
		
		self.accelerator.end_training()
	
	def save(self, dir_tag: str):
		
		# Create a directory to save the model
		save_at = os.path.join(self.args.log_dir, dir_tag)
		if not os.path.exists(save_at):
			os.makedirs(save_at)
		
		# Save Encoder
		enc = self.accelerator.unwrap_model(self.model['enc'])
		torch.save(enc.state_dict(), os.path.join(save_at, "clf_predictor.pt"))
		
		# Save Decoder
		dec = self.accelerator.unwrap_model(self.model['dec'])
		dec.save_pretrained(
			save_directory=os.path.join(save_at, "PEFT"),
			is_main_process=is_rank_0(),
		)  # In place of $ accelerator.save(model.state_dict(), path)
		
		if is_rank_0():
			print("Saved the Decoder model at:", os.path.join(save_at, "PEFT"))
			print("Saved the Encoder model at:", os.path.join(save_at, "clf_predictor.pt"))


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.

	trainer = Trainer(args, logger)
	trainer.train_loop()


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file basic_nofp16.yaml train_cvae.py --load_base_from_path ./logging/codegen-350m/Baseline_1.0/output/pytorch_model.bin --do_peft 1
	main()
