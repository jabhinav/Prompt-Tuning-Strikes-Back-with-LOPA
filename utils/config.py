import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

from utils.custom import create_dir, is_rank_0, set_dist, set_seed, log_dist
from utils.xformer import get_huggingface_path

from huggingface_hub import login


def get_config():
	
	parser = argparse.ArgumentParser()
	
	# #################################################### High-level ############################################### #
	parser.add_argument("--peft_method", type=str, default=None,
						choices=['lopa', 'pt', 'idpg', 'lora'])
	parser.add_argument("--task_name", type=str, default=None,
						choices=['mbpp', 'cruxeval_input_prediction', 'cruxeval_output_prediction'])
	parser.add_argument('--wandb_logging', type=bool, default=False)
	parser.add_argument('--project_name', type=str, default='PromptTuningModel', help="Name of the wandb project")
	parser.add_argument('--seed', type=int, default=9876, help="random seed for init.")
	
	# #################################################### Model #################################################### #
	parser.add_argument("--model_type", type=str, default=None,
						choices=["phi-2", "phi-3", "codegen-350M", "codegen2-3_7B", "deepseek-coder-1.3b-base",
								 "deepseek-coder-7b-base", "Meta-Llama-3-8B"])
	parser.add_argument("--enc_model_type", type=str, default="codesage-small",
						choices=["codebert-base", "codet5p-110m-embedding",
								 "codesage-small", "codesage-base", "codesage-large"])
	
	# #################################################### Paths #################################################### #
	parser.add_argument('--path_to_data', type=str, default='./data/MBPP/mbpp_release_v1.jsonl')  # Used by MBPP
	parser.add_argument('--load_adapter_from', type=str, default=None)  # Path to dir
	parser.add_argument('--load_base_from_path', type=str, default=None)
	parser.add_argument('--sharded_checkpoint_dir', type=str, default=None)
	parser.add_argument('--clf_predictor_path', type=str, default=None)
	parser.add_argument('--log_dir', type=str, default=None)
	
	# #################################### Prompt Tuning Parameters ################################################# #
	parser.add_argument('--num_virtual_tokens', type=int, default=10)
	parser.add_argument("--lp_rank", type=int, default=1, help="Low-Rank for matrix factorization in LOPA",
						choices=[1, 2, 4])

	# ################################################### Training ################################################## #
	parser.add_argument("--num_epochs", type=int, default=10)  # 4 for fft on crux-eval, 10 for fft on mbpp
	parser.add_argument("--per_gpu_train_batch_size", type=int, default=2)
	parser.add_argument("--lr", type=float, default=1e-3,
						help="1e-3 for PT/IDPG/Ours, 1e-4 for LoRA(r=16, alpha=32), 1e-5 for FFT")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--num_warmup_steps", type=int, default=0)
	parser.add_argument("--weight_decay", type=float, default=0.05)  # Used only by FFT
	parser.add_argument("--lr_scheduler_type", type=str, default='linear',
						choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
								 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau'])   # Used only by FFT

	# ################################################## Evaluation ################################################# #
	parser.add_argument("--num_beams", type=int, default=1)
	parser.add_argument("--do_sample", type=bool, default=True)
	parser.add_argument("--num_return_sequences", type=int, default=1)
	parser.add_argument("--num_return_sequences_per_iter", type=int, default=1)
	parser.add_argument("--temperature", type=float, default=None, choices=[0.2, 0.6, 0.8])
	parser.add_argument("--top_p", type=float, default=0.95)
	parser.add_argument("--top_k", type=float, default=-1)
	
	# ########################################### Logging Configuration ############################################# #
	parser.add_argument("--save_every", type=int, default=-1,
						help="Save model every n epochs (if > 0)")
	parser.add_argument("--save_steps", default=0, type=int,
						help="Save model every n steps (If > 0) if `save_strategy==steps`")  # Used by FFT
	parser.add_argument('--save_total_limit', default=1, type=int,
						help='total of number checkpoints to keep; only keep the latest ones')  # Used by FFT
	parser.add_argument("--log_interval", default=1, type=int,
						help="Log every X updates steps.")  # Used by FFT
	parser.add_argument("--huggingface_login_token", type=str, default='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',
						help="Hugging Face login token")
	
	# ############################################ Hardware configuration ########################################### #
	parser.add_argument("--local_rank", type=int, default=-1, help="For multi-node training: local_rank")
	parser.add_argument('--fp16', default=False, help="Whether to use 16-bit (mixed) precision "
													  "(through NVIDIA apex) instead of 32-bit")  # Used by FFT
	parser.add_argument('--path_to_ds_config', default='./zero_stage2_nofp16_config.json', type=str,
						help="path to deepspeed configuration file; set None if not using deepspeed")  # Used by FFT
	parser.add_argument("--load_in_8bit", type=bool, default=False,)
	parser.add_argument("--no_cuda", help="Avoid using CUDA when available")
	parser.add_argument("--node_index", type=int, default=-1, help="node index if multi-node running")
	parser.add_argument("--gpu_per_node", type=int, default=4, help="num of gpus per node")
	
	# # #################################### EM-specific #################################### #
	# parser.add_argument('--num_libraries', type=int, default=5)
	# parser.add_argument("--num_init_epochs", type=int, default=5)
	# parser.add_argument("--pre_num_iters", type=int, default=500)
	# parser.add_argument("--max_m_steps", type=int, default=10)
	# parser.add_argument("--init_lr", type=float, default=5e-3)
	# parser.add_argument("--prior_lr", type=float, default=5e-6)
	#
	# # #################################### VAE params #################################### #
	# parser.add_argument("--enc_lr", type=float, default=1e-3)
	# parser.add_argument("--dec_lr", type=float, default=5e-4)
	# parser.add_argument("--kl_coeff", type=float, default=1.0)
	#
	# # #################################### Others #################################### #
	# parser.add_argument("--num_test_problems", type=int, default=None, choices=[None, 100])
	# parser.add_argument("--num_train_problems", type=int, default=None, choices=[None, 100])
	# parser.add_argument("--infer_final_responsibilities", type=bool, default=False)
	
	args = parser.parse_args()
	
	# Login to the Hugging Face Hub
	if args.huggingface_login_token is not None:
		login(token=args.huggingface_login_token)
	
	# Get huggingface paths for the models
	args.model_name_or_path = get_huggingface_path(args.model_type)
	args.config_name = get_huggingface_path(args.model_type)
	args.tokenizer_name = get_huggingface_path(args.model_type)
	
	# Create a directory to store the logs
	if args.log_dir is None:
		current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
		log_dir = os.path.join('./logging', current_time)
		args.log_dir = log_dir
	create_dir(args.log_dir)
	
	# Configure logging
	if is_rank_0():
		logging.basicConfig(filename=os.path.join(args.log_dir, 'logs.txt'), filemode='w',
							format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	# Set save paths
	args.save_at = os.path.join(args.log_dir, 'PromptTuningMultiModel')
	args.save_results_at = os.path.join(args.log_dir, 'all_codes.json')
	
	# Set the distributed training
	set_dist(args)
	
	# Set the seed
	set_seed(args)
	
	# # Update the args with task-specific custom attributes
	add_task_attributes(args)
	
	# Log the config
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	log_dist(message="\n\n# ############### PEFT ############## #\n\n", level=logging.INFO, ranks=[0])
	log_dist(message=json.dumps(config, indent=4), level=logging.INFO, ranks=[0])
	
	return args, logger
	

def add_task_attributes(args):
	"""
	Following values were used to compute the results. Overwrite them if required.
	:param args:
	:return:
	"""
	
	if 'cruxeval' in args.task_name:
		# Update the max length [used by cruxeval -> 1024]
		# Note: this will override the deduction of num_virtual_tokens from max_length. So, virtual tokens will be appended
		# Useful when doing ablation on number of virtual tokens
		args.max_length = 512  # I have chosen this based on the max length of the prompt and the code generation.
		
		# Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'
		args.prefix = ""
		
		# Use chain-of-thoughts for the prompt? [Not supported for now]
		args.cot = False
		
		# Temperature for generation: Choose from 0.2, 0.8 [both used by cruxeval]
		args.temperature = 0.2
		
	elif args.task_name == 'mbpp':
		
		args.max_prompt_length = 325
		args.max_new_tokens = 256
		args.max_length = args.max_prompt_length + args.max_new_tokens
		
		# [If number of tokens in fwd pass (including virtual tokens) are kept to be = max_length] :-
		# Update the max_length and max_prompt_length by deducting the number of virtual tokens.
		args.max_length = args.max_length - args.num_virtual_tokens
		args.max_prompt_length = args.max_prompt_length - args.num_virtual_tokens
		
		# Temperature for generation:
		args.temperature = 0.6
