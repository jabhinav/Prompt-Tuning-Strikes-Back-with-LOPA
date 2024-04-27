import math
import os
import time

import accelerate
import torch
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_scheduler

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftCvaeModel
from utils.config import get_config
from utils.custom import is_rank_0
from utils.data import MBPP_Dataset_wEnc
from utils.data import CruxEval_Dataset_wEnc
from utils.model import LatentPromptAttentionGenerator as EmbeddingEncoder
from utils.xformer import load_base_model
from utils.xformer import load_tokenizer, get_huggingface_path


class CVAE(torch.nn.Module):
	
	def __init__(self, enc, dec):
		super(CVAE, self).__init__()
		self.enc = enc
		self.dec = dec
		
		# Beta
		self.config = self.enc.base.config
		
	def forward(self, batch):

		# Encode
		att_logits = self.enc(
			input_ids=batch['enc_input_ids'],
			attention_mask=batch['enc_attention_mask'],
		)
		att_weights = torch.sigmoid(att_logits)
		
		# Decode
		output = self.dec(
			latent_attention_weights=att_weights,
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			labels=batch['labels'],
			output_hidden_states=True
		)
		
		return output
	

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
			enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params = self.__count_parameters(
				self.model)
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
		self.args.device = self.accelerator.device
		
		# Finally, initialize the trackers. During init of the model we computed new arguments. Thus setting after that.
		self.init_trackers()
	
	def _init_accelerator(self):
		
		project_config = ProjectConfiguration(
			logging_dir=self.args.log_dir,
		)
		kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		
		# when using DeepSpeed, the `gradient_accumulation_steps` is properly set either
		# > from the DeepSpeed plugin/config
		# > from `accelerate launch` via `--gradient_accumulation_steps`
		# > defaulting to the passed `args.gradient_accumulation_steps` (using this + setting auto in the config file)
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
	
	def _build_dataloader(self):
		# Get the dataset
		with self.accelerator.main_process_first():
			if self.args.dataset_name == 'mbpp':
				dataset = MBPP_Dataset_wEnc(
					path_to_data=self.args.path_to_data,
					tokenizer=self.tokenizer,
					max_prompt_length=self.args.max_prompt_length,
					max_length=self.args.max_length,
					sample_problems=self.args.num_train_problems,
					mode='train',
					enc_tokenizer=self.enc_tokenizer,
				)
			elif self.args.dataset_name == 'cruxeval':
				dataset = CruxEval_Dataset_wEnc(
					tokenizer=self.tokenizer,
					max_length=self.args.max_length,
					mode='train',
					enc_tokenizer=self.enc_tokenizer,
					cruxeval_task=self.args.cruxeval_task,
					prefix=self.args.prefix,
					cot=self.args.cot,
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
	
	def _build_model(self):
		dec = self.init_decoder()  # Init first
		enc = self.init_encoder()
		model = CVAE(enc, dec)
		return model
	
	def _build_optimizer(self):
		
		# Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
		optimizer_cls = (
			torch.optim.AdamW
			if self.accelerator.state.deepspeed_plugin is None
			   or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
			else accelerate.utils.DummyOptim
		)
		optimizer = optimizer_cls(self.model.parameters(), lr=self.args.lr)
		
		return optimizer
	
	def _build_scheduler(self):
		
		# Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
		if (
				self.accelerator.state.deepspeed_plugin is None
				or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
		):
			lr_scheduler = get_scheduler(
				name="constant",
				optimizer=self.optimizer,
				num_warmup_steps=self.args.num_warmup_steps,
				num_training_steps=self.args.max_train_steps,
			)
		else:
			lr_scheduler = accelerate.utils.DummyScheduler(
				self.optimizer,
				total_num_steps=self.args.max_train_steps,
				warmup_num_steps=self.args.num_warmup_steps
			)
		return lr_scheduler

	
	def _accelerator_prepare(self):
		
		self.train_dataloader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.train_dataloader, self.model, self.optimizer, self.scheduler)
	
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
		self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.total_batch_size}")
		self.logger.info(f"  Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}")
		self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
		self.logger.info("\n")
		
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"{self.args.dataset_name}/{self.args.model_type}_cvae"}}
			)
	
	@staticmethod
	def __count_parameters(model):
		enc_trainable_params = sum(p.numel() for p in model.enc.parameters() if p.requires_grad)
		enc_all_params = sum(p.numel() for p in model.enc.parameters())
		dec_trainable_params, dec_all_params = model.dec.get_nb_trainable_parameters()
		return enc_trainable_params, enc_all_params, dec_trainable_params, dec_all_params
	
	def _train_step(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		
		with self.accelerator.accumulate(self.model):
			output = self.model(batch)
			
			# Loss function = -log(p(x|z))
			reconstruction_loss = output.loss
			
			# Total loss
			total_loss = reconstruction_loss
			
			# BP and Grad Updated
			self.accelerator.backward(total_loss)
			self.optimizer.step()
			self.scheduler.step()
			self.optimizer.zero_grad()
			
			if self.accelerator.sync_gradients:
				# Updating the current step under the accumulate context manager takes care of everything
				self.step += 1
		
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
		
		# Set the model to train mode
		self.model.train()
		
		epoch_losses: dict = {}

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
			train_losses, train_stats = self._train_step(batch)
				
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
						"Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
					},
					step=self.step,
				)
		
		self.accelerator.wait_for_everyone()
		
		# Compute the average losses for the epoch
		for key in epoch_losses.keys():
			epoch_losses[key] = (
					epoch_losses[key] / len(self.train_dataloader) * self.args.gradient_accumulation_steps
			)
		
		return epoch_losses
	
	def train_loop(self):
		r"""Training loop. The public entry of training process."""
		
		# # For Debugging
		# if self.accelerator.is_main_process:
		# 	self.save("init")
		
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
		
		model = self.accelerator.unwrap_model(self.model)
		
		# Save Encoder
		torch.save(model.enc.state_dict(), os.path.join(save_at, "clf_predictor.pt"))
		
		# Save Decoder
		model.dec.save_pretrained(
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
	# To run with accelerate, $ accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train_cvae.py --load_base_from_path ./logging/codegen-350m/Baseline_1.0/output/pytorch_model.bin --do_peft 1
	main()
