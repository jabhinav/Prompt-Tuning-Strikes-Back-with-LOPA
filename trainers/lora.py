import os

import torch
from peft import LoraConfig, get_peft_model, PeftModel

from trainers.base import BaseTrainer
from utils.custom import is_rank_0
from utils.xformer import load_base_model, LORA_IA3_TARGET_MODULES


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
		
		# # Uncomment this to identify target modules for LoRA
		# print(model.state_dict().keys())
		# sys.exit(-1)
		
		# Get the config
		lora_config = LoraConfig(
			r=self.args.lora_dim,
			lora_alpha=self.args.lora_alpha,
			target_modules=LORA_IA3_TARGET_MODULES[args.model_type]["target_modules_lora"],
			lora_dropout=self.args.lora_dropout,
			bias="none",
			# modules_to_save=["classifier"],  # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
			fan_in_fan_out=True if 'gpt2' in args.model_type else False,  # True if the model has fan_in_fan_out
		)
		
		if args.load_adapter_from is not None:
			# Load the model adapters - in place
			model = PeftModel.from_pretrained(
				model=model,
				model_id=args.load_adapter_from,  # Must be a directory containing the model files
			)
			msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
			if is_rank_0():
				print(msg)
		else:
			# Initialize the model adapters
			model = get_peft_model(model, lora_config)
		
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
				init_kwargs={"wandb": {"name": f"{self.args.task_name}/{self.args.model_type}/lora"}}
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
