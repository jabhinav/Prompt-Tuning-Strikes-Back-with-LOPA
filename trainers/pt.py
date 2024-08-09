import os

import torch

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel
from trainers.base import BaseTrainer
from utils.custom import is_rank_0
from utils.xformer import load_base_model


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
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
		pt_config = PromptTuningConfig(
			task_type=TaskType.CAUSAL_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
		)
		
		if args.load_adapter_from is not None:
			# Load the model adapters - in place
			model = PeftModel.from_pretrained(
				model=model,
				model_id=args.load_adapter_from,  # Must be a directory containing the model files
				config=pt_config,
			)
			msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
			if is_rank_0():
				print(msg)
		else:
			# Initialize the model adapters
			model = get_peft_model(model, pt_config)
		
		# This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
		self.args.total_virtual_tokens = self.args.num_virtual_tokens * pt_config.num_transformer_submodules
		self.args.word_embedding_dim = pt_config.token_dim
		
		return model
	
	def _build_model(self):
		model = self.init_decoder()
		return model
	
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"{self.args.task_name}/{self.args.model_type}/pt"}}
			)
	
	def count_parameters(self):
		dec_trainable_params, dec_all_params = self.model.get_nb_trainable_parameters()
		return None, None, dec_trainable_params, dec_all_params
	
	def _train_step(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		
		with self.accelerator.accumulate(self.model):
			output = self.model(
				input_ids=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				labels=batch["labels"],
				output_hidden_states=True
			)
			
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
		}
	
	def _eval_step(self, batch):
		with self.accelerator.accumulate(self.model):
			output = self.model(
				input_ids=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				labels=batch["labels"],
				output_hidden_states=True
			)
			reconstruction_loss = output.loss
		return reconstruction_loss
	
	def save(self, dir_tag: str):
		
		# Create a directory to save the model
		save_at = os.path.join(self.args.log_dir, dir_tag)
		if not os.path.exists(save_at):
			os.makedirs(save_at)
		
		# Save Decoder
		model = self.accelerator.unwrap_model(self.model)
		model.save_pretrained(
			save_directory=os.path.join(save_at, "PEFT"),
			is_main_process=is_rank_0(),
		)
		
		if is_rank_0():
			print("Saved the Decoder model at:", os.path.join(save_at, "PEFT"))
