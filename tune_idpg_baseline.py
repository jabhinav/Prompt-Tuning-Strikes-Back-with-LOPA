import os

import torch
from accelerate.logging import MultiProcessAdapter

from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftIDPGModel
from utils.config import get_config
from utils.custom import is_rank_0
from utils.model import IDPGSoftPromptGenerator as EmbeddingEncoder, IDPG
from utils.trainer import BaseTrainer
from utils.xformer import load_base_model


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
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
			task_type=TaskType.IDPG_CAUSAL_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
		)
		
		if args.load_adapter_from is not None:
			# Load the model adapters - in place
			model = PeftIDPGModel.from_pretrained(
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
		
		dec = self.init_decoder()  # Init first
		enc = self.init_encoder()
		model = IDPG(enc, dec)
		
		return model
	
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"{self.args.dataset_name}/{self.args.model_type}_idpg"}}
			)
	
	def count_parameters(self):
		enc_trainable_params = sum(p.numel() for p in self.model.enc.parameters() if p.requires_grad)
		enc_all_params = sum(p.numel() for p in self.model.enc.parameters())
		return enc_trainable_params, enc_all_params, None, None
	
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
		}
	
	def save(self, dir_tag: str):
		
		# Create a directory to save the model
		save_at = os.path.join(self.args.log_dir, dir_tag)
		if not os.path.exists(save_at):
			os.makedirs(save_at)
		
		# Save Encoder
		enc = self.accelerator.unwrap_model(self.model.enc)
		state_dict = enc.state_dict()
		# Remove the enc.base layers before saving [IDPG-specific]
		state_dict = {k: v for k, v in state_dict.items() if "base" not in k}
		torch.save(state_dict, os.path.join(save_at, "clf_predictor.pt"))
		del state_dict
		
		# Save Decoder [won't be used, saving for consistency]
		dec = self.accelerator.unwrap_model(self.model.dec)
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
	
	# To force the encoder to be same as the decoder
	# args.enc_model_type = args.model_type
	
	trainer = Trainer(args, logger)
	trainer.train_loop()


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml tune_idpg_baseline.py
	main()
