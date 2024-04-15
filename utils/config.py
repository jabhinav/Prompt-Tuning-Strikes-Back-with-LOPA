import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

from utils.custom import create_dir, is_rank_0, set_dist, set_seed, log_dist
from utils.xformer import get_huggingface_path


def get_config():
	# Create a directory to store the logs
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
	model_type = "phi-2"  # codegen-350M, phi-2, codegen2-3_7B, deepseek-coder-1.3b-base, deepseek-coder-7b-base
	huggingface_path = get_huggingface_path(model_type)
	
	parser = argparse.ArgumentParser()
	
	# High-level
	parser.add_argument('--wandb_logging', type=bool, default=True)
	parser.add_argument('--project_name', type=str, default='PromptTuningModel')
	parser.add_argument('--do_peft', type=int, default=None)
	parser.add_argument('--seed', type=int, default=9876, help="random seed for initialization")
	
	# Paths
	parser.add_argument('--path_to_data', type=str, default='./data/MBPP/mbpp_release_v1.jsonl')
	parser.add_argument('--save_at', type=str, default=log_dir + '/PromptTuningMultiModel')
	parser.add_argument('--load_adapter_from', type=str, default=None)  # Path to dir
	parser.add_argument('--load_base_from_path', type=str, default=None)
	parser.add_argument('--clf_predictor_path', type=str, default=None)
	
	# Prompt Tuning Parameters
	parser.add_argument('--num_libraries', type=int, default=5)
	parser.add_argument('--num_virtual_tokens', type=int, default=10)
	parser.add_argument('--max_prompt_length', type=int, default=325)  # Max 384
	parser.add_argument('--max_length', type=int, default=325+256)  # Max 384+512
	parser.add_argument("--max_new_tokens", type=int, default=256)
	
	# Model
	parser.add_argument("--model_type", type=str, default=model_type)
	parser.add_argument("--model_name_or_path", type=str, default=huggingface_path)
	parser.add_argument("--config_name", type=str, default=huggingface_path)
	parser.add_argument("--tokenizer_name", type=str, default=huggingface_path)
	
	# Training
	parser.add_argument("--num_epochs", type=int, default=10)
	parser.add_argument("--per_gpu_train_batch_size", type=int, default=2)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--ent_coeff", type=float, default=0.0)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
	parser.add_argument("--num_warmup_steps", type=int, default=0)
	parser.add_argument("--save_every", type=int, default=4)
	
	# EM-specific
	parser.add_argument("--num_init_epochs", type=int, default=5)
	parser.add_argument("--pre_num_iters", type=int, default=500)
	parser.add_argument("--max_m_steps", type=int, default=10)
	parser.add_argument("--init_lr", type=float, default=5e-3)
	parser.add_argument("--prior_lr", type=float, default=5e-6)
	
	# VAE params
	parser.add_argument("--enc_model_type", type=str, default="codebert-base")  # "codebert-base", "codet5p-110m-embedding"
	parser.add_argument("--enc_lr", type=float, default=1e-3)
	parser.add_argument("--dec_lr", type=float, default=5e-4)
	parser.add_argument("--kl_coeff", type=float, default=10.0)
	
	# Others
	parser.add_argument("--num_test_problems", type=int, default=None, choices=[None, 100])
	parser.add_argument("--num_train_problems", type=int, default=None, choices=[None, 100])
	parser.add_argument("--infer_final_responsibilities", type=bool, default=False)
	
	# Evaluation
	parser.add_argument("--save_results_at", type=str, default=os.path.join(log_dir, 'all_codes.json'))
	parser.add_argument("--num_beams", type=int, default=1)
	parser.add_argument("--do_sample", type=bool, default=True)
	parser.add_argument("--num_return_sequences", type=int, default=1)
	parser.add_argument("--num_return_sequences_per_iter", type=int, default=1)
	parser.add_argument("--temperature", type=float, default=0.6)
	parser.add_argument("--top_p", type=float, default=0.95)
	
	# Debugging
	parser.add_argument('--db', default=False,
						help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')
	
	# Hardware configuration
	parser.add_argument("--load_in_8bit", type=bool, default=False,)
	parser.add_argument("--no_cuda", help="Avoid using CUDA when available")
	parser.add_argument("--local_rank", type=int, default=-1,
						help="For distributed training (multi-node): local_rank")
	parser.add_argument("--node_index", type=int, default=-1, help="node index if multi-node running")
	parser.add_argument("--gpu_per_node", type=int, default=4, help="num of gpus per node")
	
	args = parser.parse_args()
	
	args.log_dir = log_dir
	
	# Update the max_length and max_prompt_length by deducting the number of virtual tokens
	args.max_length = args.max_length - args.num_virtual_tokens
	args.max_prompt_length = args.max_prompt_length - args.num_virtual_tokens
	
	set_dist(args)
	set_seed(args)
	
	update_args_with_custom_attributes(args)
	
	# Log the config
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	
	log_dist(message="\n\n# ############### PEFT ############## #\n\n", level=logging.INFO, ranks=[0])
	log_dist(message=json.dumps(config, indent=4), level=logging.INFO, ranks=[0])
	
	return args, logger


def update_args_with_custom_attributes(args):
	# [[[HERE]]] For using training data split-2
	args.finer_train_split = 1.0
	args.use_train_first_half = True
	
	# [[[HERE]]] Only used by prior
	args.path_to_train_prior_labels = './logging/train_gt_instance.json'
	
