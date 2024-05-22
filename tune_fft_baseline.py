import os

import torch
import transformers
from accelerate.logging import MultiProcessAdapter
from transformers.modeling_utils import unwrap_model

import logging
from utils.CustomTensorboardCallback import CustomTensorBoardCallback
from utils.config import get_config
from utils.custom import is_rank_0, log_dist
# from utils.data import MBPP_Dataset as CustomDataset
from utils.data import MBPP_Dataset_wEnc, CruxEval_Dataset_wEnc
from utils.model import load_base_model
from utils.xformer import load_tokenizer, get_huggingface_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def learn(args, logger):
	# if is_rank_0():
	# 	print(f"\n\nStarting training!! (Using train data split {args.finer_train_split}, First Half: {args.use_train_first_half})\n\n")
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))  # Dummy
	
	# Get the dataset
	if args.dataset_name == 'mbpp':
		dataset = MBPP_Dataset_wEnc(
			path_to_data=args.path_to_data,
			tokenizer=tokenizer,
			max_prompt_length=args.max_prompt_length,
			max_length=args.max_length,
			mode='train',
			enc_tokenizer=enc_tokenizer,
		)
	elif args.dataset_name == 'cruxeval':
		dataset = CruxEval_Dataset_wEnc(
			tokenizer=tokenizer,
			max_length=args.max_length,
			mode='train',
			enc_tokenizer=enc_tokenizer,
			cruxeval_task=args.cruxeval_task,
			prefix=args.prefix,
			cot=args.cot,
		)
	else:
		raise ValueError(f"Invalid dataset name: {args.dataset_name}")
	
	# dataset = CustomDataset(
	# 	path_to_data=args.path_to_data,
	# 	tokenizer=tokenizer,
	# 	max_prompt_length=args.max_prompt_length,
	# 	max_length=args.max_length,
	# 	sample_problems=args.num_train_problems,
	# 	mode='train',
	#
	# 	# Uncomment to use a finer split of the training data to tune the baseline
	# 	finer_split=args.finer_train_split,
	# 	use_first_half=args.use_train_first_half
	# )
	# dataset.return_dict = True
	
	# # Model Loading ########################################################
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# # Load Checkpoint ########################################################
	if args.load_base_from_path is not None:
		# We load the model state dict on the CPU to avoid an OOM error.
		loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		message = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		if is_rank_0():
			logger.info(message)
			print(message)
	
	# # Start training ########################################################
	if is_rank_0():
		print("\n\nStarting main loop!!")
	
	training_args = transformers.TrainingArguments(
		
		output_dir=args.save_at,
		overwrite_output_dir=True,
		
		do_train=True,
		do_eval=False,  # Will be set to `True` if `evaluation_strategy` is different from `no`
		do_predict=False,
		evaluation_strategy='no',  # 'no', 'steps' (eval every `eval_steps`), 'epoch' (eval every end of epoch)
		eval_steps=0,
		
		num_train_epochs=args.num_epochs,
		per_device_train_batch_size=args.per_gpu_train_batch_size,  # comment out gives error
		gradient_accumulation_steps=args.gradient_accumulation_steps,  # comment out gives error
		
		learning_rate=args.lr,
		weight_decay=args.weight_decay,
		lr_scheduler_type=args.lr_scheduler_type,
		warmup_steps=args.num_warmup_steps,  # default: 0, can also provide warmup_ratio
		warmup_ratio=0.1,
		
		logging_dir=args.log_dir,
		logging_strategy='steps',
		logging_first_step=True,
		logging_steps=args.log_interval,
		
		# save_steps=args.save_steps,
		# save_total_limit=args.save_total_limit,
		save_total_limit=None,
		save_strategy='no',  # 'steps', 'epoch', 'no'  # If set to no, save manually using `trainer.save_model()`
		# save_only_model=True,
		save_safetensors=False,  # Set to True to save the safe tensors else save the model state dict as .pt file
		
		report_to=['wandb'],
		run_name=f'{args.dataset_name}/{args.model_type}_fft',  # name for the wandb run
		
		dataloader_drop_last=False,
		dataloader_num_workers=0 if args.db else 8,
		
		local_rank=args.local_rank,
		deepspeed=args.path_to_ds_config,
		fp16=args.fp16,
	)
	
	trainer = transformers.Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
	)
	
	trainer.remove_callback(transformers.integrations.TensorBoardCallback)
	trainer.add_callback(CustomTensorBoardCallback())
	
	trainer.train()
	
	# Save the model [My way]
	log_dist(
		message="Saving model at {}".format(args.save_at),
		level=logging.INFO,
		ranks=[0]
	)
	
	trainer.save_state()  # Save "trainer_state.json"
	trainer.save_model()  # Save the model (will be done in the `save_at` directory provided in `training_args`)


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	# Train model
	learn(args, logger)


if __name__ == "__main__":
	"""
	For complete guide:
	https://huggingface.co/docs/transformers/deepspeed
	FFT large models is expensive
	On anton (16GBx4):
	$ deepspeed tune_fft_baseline.py # with stage 2 works fine but without fp16. Even with stage 2 1.3b goes OOM
	On A100 (40GBx2):
	$ deepspeed tune_fft_baseline.py --path_to_ds_config ./zero_stage3_config.json --fp16 True --gradient_accumulation_steps 2
	Now stage-3 with fp-16 works. Memory usage is around 8GB per GPU. Amazing!
	"""
	main()
