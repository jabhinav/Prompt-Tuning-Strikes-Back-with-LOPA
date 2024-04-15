import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
import transformers
from transformers.modeling_utils import unwrap_model

import logging
from utils.CustomTensorboardCallback import CustomTensorBoardCallback
from utils.custom import is_rank_0, log_dist, create_dir, set_dist, set_seed
from utils.data import MBPP_Dataset as CustomDataset
from utils.model import load_base_model
from utils.xformer import load_tokenizer, get_huggingface_path


def get_config():
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join('./logging', current_time)
	create_dir(log_dir)
	
	# Configure logging
	if is_rank_0():
		logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
							format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Define the parameters
	model_type = "codegen2-3_7B"  # codegen-350M, codegen2-1B, codegen2-3_7B, deepseek-coder-1.3b-base
	huggingface_path = get_huggingface_path(model_type)
	
	parser = argparse.ArgumentParser()

	# ############################################### High-level ################################################# #
	parser.add_argument("--wandb_run_name", type=str, default=f'MBPP_{model_type}')
	parser.add_argument('--project_name', type=str, default='PromptTuningModel')
	parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
	
	# ######################################### Tokenizer Configuration ########################################## #
	parser.add_argument('--max_prompt_length', type=int, default=325)  # Max 384
	parser.add_argument('--max_length', type=int, default=325+256)  # Max 384+512
	
	# ###################################### Path to directories [Required] ####################################### #
	parser.add_argument("--log_dir", default=log_dir, type=str)
	parser.add_argument('--path_to_data', type=str, default='./data/MBPP/mbpp_release_v1.jsonl')
	parser.add_argument('--save_at', type=str, default=log_dir + '/output')
	parser.add_argument('--load_from_path', type=str, default=None)
	parser.add_argument('--path_to_ds_config', default='./ds_config_for_tuning.json', type=str,
						help="path to deepspeed configuration file; set None if not using deepspeed")
	
	# ########################################### Model Configuration ############################################ #
	parser.add_argument("--model_type", default=model_type, type=str)
	parser.add_argument("--model_name_or_path", type=str, default=huggingface_path)
	parser.add_argument("--config_name", type=str, default=huggingface_path)
	parser.add_argument("--tokenizer_name", type=str, default=huggingface_path)
	
	# #################################### Training-Optimizer Configuration #################################### #
	parser.add_argument("--num_epochs", type=int, default=10)
	parser.add_argument("--per_gpu_train_batch_size", type=int, default=1)  # 2 works for codegen-350M
	parser.add_argument("--lr", type=float, default=1e-5)  # (5e-5, codegen-350M)
	parser.add_argument('--gradient_accumulation_steps', type=int, default=8)  # 8 works for codegen-350M
	parser.add_argument("--warmup_steps", type=int, default=100)
	parser.add_argument("--weight_decay", type=float, default=0.05)
	parser.add_argument("--lr_scheduler_type", type=str, default='cosine',
						choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
								 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau'])
	
	# #################################### Logging Configuration ############################################# #
	parser.add_argument("--save_steps", default=0, type=int,
						help="Save model every n steps (If > 0) if `save_strategy==steps`")
	parser.add_argument('--save_total_limit', default=2, type=int,
						help='total of number checkpoints to keep; only keep the latest ones')
	parser.add_argument("--log_interval", default=1, type=int,
						help="Log every X updates steps.")

	# #################################### Hardware Configuration ############################################# #
	parser.add_argument("--load_in_8bit", type=bool, default=False)
	parser.add_argument("--no_cuda",
						help="Avoid using CUDA when available")
	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training (multi-node): local_rank")
	parser.add_argument('--db', default=False,
						help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')
	parser.add_argument("--node_index", type=int, default=-1,
						help="node index if multi-node running")
	parser.add_argument("--gpu_per_node", type=int, default=4,
						help="num of gpus per node")
	
	# ############################################ Others ##################################################### #
	parser.add_argument("--num_train_problems", type=int, default=None, choices=[None, 100])
	
	args = parser.parse_args()
	
	# Create Directories
	create_dir(args.save_at)
	
	set_dist(args)
	set_seed(args)
	
	update_args_with_custom_attributes(args)
	
	# Log the config
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	log_dist(message="\n\n# ############### Tune Baseline ############## #\n\n", level=logging.INFO, ranks=[0])
	log_dist(message=json.dumps(config, indent=4), level=logging.INFO, ranks=[0])
	
	return args, logger


def update_args_with_custom_attributes(args):
	# [[[HERE]]] For using training data split-1
	args.finer_train_split = 1.0
	args.use_train_first_half = True


def learn(args, logger):
	
	if is_rank_0():
		print(f"\n\nStarting training!! (Using train data split {args.finer_train_split}, First Half: {args.use_train_first_half})\n\n")
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
	train_data = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode='train',
		
		# Uncomment to use a finer split of the training data to tune the baseline
		finer_split=args.finer_train_split,
		use_first_half=args.use_train_first_half
	)
	train_data.return_dict = True
	
	# # Model Loading ########################################################
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# # Load Checkpoint ########################################################
	if args.load_from_path is not None:
		# We load the model state dict on the CPU to avoid an OOM error.
		loaded_state_dict = torch.load(args.load_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		message = "Loaded model checkpoint from path: {}".format(args.load_from_path)
		if is_rank_0():
			logger.info(message)
			print(message)
	
	# # Start training ########################################################
	if is_rank_0():
		print("\n\nStarting main loop!!")
	
	training_args = transformers.TrainingArguments(
		
		output_dir=args.save_at,
		overwrite_output_dir=False,
		
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
		warmup_steps=args.warmup_steps,  # default: 0, can also provide warmup_ratio
		# warmup_ratio=0.1,
		
		logging_dir=args.log_dir,
		logging_strategy='steps',
		logging_first_step=True,
		logging_steps=args.log_interval,
		
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		save_strategy='steps',
		
		report_to=['wandb'],
		run_name=args.wandb_run_name,  # name for the wandb run
		
		dataloader_drop_last=False,
		dataloader_num_workers=0 if args.db else 8,
		
		local_rank=args.local_rank,
		deepspeed=args.path_to_ds_config,
		fp16=args.fp16,
	)
	
	trainer = transformers.Trainer(
		model=model,
		args=training_args,
		train_dataset=train_data,
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
	if is_rank_0():
		print("Saving model at {}".format(args.save_at))
	torch.save(unwrap_model(model).state_dict(), os.path.join(args.save_at, "pytorch_model.bin"))


def main():
	
	args, logger = get_config()
	
	# Train model
	learn(args, logger)


if __name__ == "__main__":
	# To Use DeepSpeed (multi-gpu): $ deepspeed tune_fft_baseline.py
	main()
