import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from transformers.modeling_utils import load_sharded_checkpoint

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from utils.config import get_config
from utils.eval import save_predictions_mbxp_format, decode_predictions
from torch.utils.data.dataloader import DataLoader
from utils.data import MBPP_Dataset_wEnc as CustomDataset
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def load_model(args, logger, accelerator):
	
	# Get the model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# Larger models are sharded
	if args.sharded_checkpoint_dir is not None:
		# Ref: https://huggingface.co/docs/transformers/big_models
		load_sharded_checkpoint(model, args.sharded_checkpoint_dir)
		msg = "[INFO] Loaded the sharded checkpoint from: {}".format(args.sharded_checkpoint_dir)
		logger.info(msg)
		if accelerator.is_local_main_process:
			print(msg)
	
	# Load checkpoint
	if args.load_base_from_path is not None:
		# We load the model state dict on the CPU to avoid an OOM error.
		loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		message = "[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		logger.info(message)
		if accelerator.is_local_main_process:
			print(message)
	
	if args.load_adapter_from is not None:
		# Get the config
		peft_config = PromptTuningConfig(
			task_type=TaskType.CAUSAL_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
		)
		
		# Load the model adapters - in place
		model = PeftModel.from_pretrained(
			model=model,
			model_id=args.load_adapter_from,  # Must be a directory containing the model files
			config=peft_config,
		)
		msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
		logger.info(msg)
		if accelerator.is_local_main_process:
			print(msg)
	
	return model


@torch.no_grad()
def evaluate(args, logger):
	
	transformers.logging.set_verbosity_error()
	accelerator = Accelerator()
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))  # Dummy
	
	# Get the dataset
	dataset = CustomDataset(
			path_to_data=args.path_to_data,
			tokenizer=tokenizer,
			max_prompt_length=args.max_prompt_length,
			max_length=args.max_length,
			mode='test',
			enc_tokenizer=enc_tokenizer,
			
			# # Uncomment to use a finer split of the training data to evaluate
			# finer_split=0.50,
			# use_first_half=False
		)
	
	# # Leave this as is to only read prompt for any type of data
	ds_loader = DataLoader(dataset, batch_size=1)
	
	# Get the model
	model = load_model(args, logger, accelerator)
	model.eval()
	
	if args.load_in_8bit:
		# model.to() is not supported for 8bit and 4bit models
		model, ds_loader = accelerator.prepare(model, ds_loader)
	else:
		# we only wrap data loader to avoid extra memory occupation
		model = model.to(accelerator.device)
		ds_loader = accelerator.prepare(ds_loader)
	
	# Predict for each sample output by each library
	# oracle_output: Dict[str, Dict[str, List[str]]] = {}
	
	# Prepare the generation kwargs
	kwargs = {
		"max_new_tokens": args.max_new_tokens,
		"do_sample": args.do_sample,
		"num_beams": args.num_beams,
		"early_stopping": True if args.num_beams > 1 and not args.do_sample else False,
		"temperature": args.temperature if args.do_sample else 1.0,
		"top_p": args.top_p if args.do_sample else 1.0,
		"num_return_sequences": args.num_return_sequences,
	}
	
	gen_token_dict = defaultdict(list)  # dict of list of generated tokens

	for step, batch in tqdm(
			enumerate(ds_loader),
			total=math.ceil(len(dataset) / accelerator.num_processes),
			desc="Generating Predictions",
			colour="GREEN",
			# leave=False,
			dynamic_ncols=True,
			smoothing=0.04,
			disable=not accelerator.is_main_process,
	):
		
		is_wrapped = args.load_in_8bit
		if is_wrapped:
			# 8bit and 4bit models are wrapped in accelerator
			generated_tokens = accelerator.unwrap_model(model).generate(
				input_ids=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				**kwargs
			)
		else:
			generated_tokens = model.generate(
				input_ids=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				**kwargs
			)
		
		# each task is generated batch_size times
		generated_tasks = batch["task_id"].repeat(args.num_return_sequences)
		generated_tokens = accelerator.pad_across_processes(
			generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
		)
		generated_tokens, generated_tasks = accelerator.gather(
			(generated_tokens, generated_tasks)
		)
		generated_tokens = generated_tokens.cpu().numpy()
		generated_tasks = generated_tasks.cpu().numpy()
		
		for sample, generated_tokens in zip(generated_tasks, generated_tokens):
			gen_token_dict[sample].append(generated_tokens)
	
	code_gens: List[List[Optional[str]]] = decode_predictions(args, gen_token_dict, tokenizer, dataset)
	
	if accelerator.is_main_process:
		# Map back the task ids to the original ids
		oracle_output = {dataset.ids[k]: v for k, v in enumerate(code_gens)}
		
		# Save the output
		with open(args.save_results_at, 'w') as f:
			json.dump(oracle_output, f, indent=4)
		
		save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	# # Debug
	# args.do_peft = 1
	# args.load_base_from_path = './logging/Baseline_0.50/output/pytorch_model.bin'
	# args.load_adapter_from = './logging/PEFT_Oracle_0.50_0.50_20ep/PromptTuningMultiModel'
	
	evaluate(args, logger)


if __name__ == '__main__':
	main()
