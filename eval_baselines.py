import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from utils.config import get_config
from utils.eval import save_predictions_mbxp_format
from utils.data import MBPP_Dataset_wEnc as CustomDataset
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def load_decoder(args, logger):
	
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
		message = "[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		logger.info(message)
		print(message)
	
	if args.do_peft:
		
		if not os.path.exists(args.load_adapter_from):
			logger.error("Please specify the correct path to load the model adapters from")
			raise ValueError("Please specify the correct path to load the model adapters from")
		
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
		print(msg)
	
	return model


@torch.no_grad()
def evaluate(args, logger):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))  # Dummy
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test',
		enc_tokenizer=enc_tokenizer,
		
		# # Uncomment to use a finer split of the training data to evaluate
		# finer_split=0.50,
		# use_first_half=False
	)
	
	# # Leave this as is to only read prompt for any type of data
	dataset.mode = 'test'
	
	# Get the model
	model = load_decoder(args, logger)
	model.to(args.device)
	model.eval()
	
	# Predict for each sample output by each library
	num_loops = int(args.num_return_sequences / args.num_return_sequences_per_iter)
	oracle_output: Dict[str, Dict[str, List[str]]] = {}
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) if tensor is not None else None for tensor in sample)
		_, _, input_ids, attention_mask = sample
		
		all_responses: List[str] = []
		try:
			
			for _ in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
				top_responses = model.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					max_new_tokens=args.max_new_tokens,
					do_sample=args.do_sample,
					num_beams=args.num_beams,
					early_stopping=True if args.num_beams > 1 and not args.do_sample else False,
					temperature=args.temperature if args.do_sample else 1.0,
					top_p=args.top_p if args.do_sample else 1.0,
					num_return_sequences=args.num_return_sequences_per_iter,
				)
				
				top_responses = top_responses.detach().cpu().numpy().tolist()
				
				if 't5' not in args.model_type:  # For T5, we don't have to remove the prompt
					top_responses = [resp[args.max_prompt_length:] for resp in top_responses]
					top_responses = [tokenizer.decode(resp, skip_special_tokens=False) for resp in top_responses]
					# Split the response at the first occurrence of the end of text token.
					# This works since we append the eos token to responses and make the model predict it
					# Also useful to not consider any text after the first occurrence of the eos token
					top_responses = [resp.split(tokenizer.eos_token)[0] for resp in top_responses]
				
				else:
					top_responses = [tokenizer.decode(resp, skip_special_tokens=True) for resp in top_responses]
				
				all_responses.extend(top_responses)
		
		except Exception as e:
			if isinstance(e, UnboundLocalError) and str(
					e) == "local variable 'next_tokens' referenced before assignment":
				# See https://github.com/huggingface/transformers/issues/5118
				logger.exception("Problem text was > specified tokens, so cannot do generation")
				logger.info(e)
				raise e
			else:
				logger.exception("Unexpected exception in generating solution")
				logger.info(e)
				raise e
		
		oracle_output[dataset.ids[index]] = all_responses
	
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(oracle_output, f, indent=4)
	
	save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')


def main():
	args, logger = get_config()
	
	# # Debug
	# args.do_peft = 1
	# args.load_base_from_path = './logging/Baseline_0.50/output/pytorch_model.bin'
	# args.load_adapter_from = './logging/PEFT_Oracle_0.50_0.50_20ep/PromptTuningMultiModel'
	
	evaluate(args, logger)


if __name__ == '__main__':
	main()
