import json
from collections import OrderedDict
from typing import Dict, List

import torch
from tqdm import tqdm

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftMultiModel
from utils.model import get_response_log_probs
from utils.custom import save_best_lib_predictions_mbxp_format, save_predictions_mbxp_format
from utils.config import get_config
from utils.data import MBPP_Dataset as CustomDataset
from utils.xformer import load_tokenizer, load_base_model


@torch.no_grad()
def evaluate(args, logger):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.MULTI_CAUSAL_LM,  # CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
		# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
		# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
		num_init_clusters=args.num_libraries,  # My custom field
	)
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test'
	)
	
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
		message = "Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		logger.info(message)
		print(message)
	
	# Update the model's padding token id for open-ended generation
	if 't5' not in args.model_type and model.config.pad_token_id is None:
		model.config.pad_token_id = tokenizer.pad_token_id
	
	if args.do_peft:
		
		if args.load_adapter_from is None:
			logger.error("Please specify the path to load the model adapters from")
			raise ValueError("Please specify the path to load the model adapters from")
		
		# Load the model adapters - in place
		model = PeftMultiModel.from_pretrained(
			model=model,
			model_id=args.load_adapter_from,  # Must be a directory containing the model files
			config=peft_config,
		)
		msg = "Loaded the model adapters from: {}".format(args.load_adapter_from)
		logger.info(msg)
		print(msg)
	
	# GPU-ize the model
	model.to(args.device)
	
	# Predict for each sample output by each library
	num_loops = int(args.num_return_sequences / args.num_return_sequences_per_iter)
	
	oracle_output: Dict[str, Dict[str, List[str]]] = {}
	lib_mapping:  Dict[str, int] = {}
	
	model.eval()
	
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		input_ids, attention_mask = sample
		
		# ############################## Add logic for identifying the library index ############################## #
		chosen_lib_idx = None
		if args.do_peft:
			chosen_lib_idx = get_library_index(args, input_ids, attention_mask, tokenizer, model)
			
		
		all_library_predictions: Dict[str, List[str]] = OrderedDict()
		for k in range(args.num_libraries):
			
			# Set the library index
			if args.do_peft:
				model.library_idx = k
			
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
					top_responses = [resp[sample[0].shape[1]:] for resp in top_responses]
					top_responses = [tokenizer.decode(resp, skip_special_tokens=False) for resp in top_responses]
					# Split the response at the first occurrence of the end of text token.
					# This works since we append the eos token to responses and make the model predict it
					# Also useful to not consider any text after the first occurrence of the eos token
					top_responses = [resp.split(tokenizer.eos_token)[0] for resp in top_responses]
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
				
				# # Default to empty string on errors
				# prediction = ""

			# # For APPS
			# if len(prediction):
			# 	prediction = prediction.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
			
			all_library_predictions[f'lib_{k}'] = all_responses
			
		oracle_output[dataset.ids[index]] = all_library_predictions
		
		if chosen_lib_idx is not None:
			lib_mapping[dataset.ids[index]] = int(chosen_lib_idx)
			
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(oracle_output, f, indent=4)
	
	# Save the output for the best chosen libraries
	if lib_mapping:
		save_best_lib_predictions_mbxp_format(args, oracle_output, lib_mapping, lang='python', d_type='MBPP')
	
	save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')


@torch.no_grad()
def get_library_index(args, prompt, prompt_mask, tokenizer, model):
	"""
	:param args:
	:param prompt: (batch_size, seq_len)
	:param prompt_mask: (batch_size, seq_len)
	:param tokenizer:
	:param model:
	:return: library_idx: int
	"""
	
	max_likelihood = -float('inf')
	library_idx = None
	
	# Copy the prompt into the response tensor
	response = prompt.clone()
	response[response == tokenizer.pad_token_id] = -100  # Replace pad_token_id with labels [=-100]
	response_mask = response.ne(-100)
	
	for k in range(args.num_libraries):
		
		# Likelihood of the prompt coming from the latent prompt of library k
		prompt_log_prob = get_response_log_probs(args, (prompt, prompt_mask, response, response_mask), tokenizer, model, k)
		
		# Update the library index
		if prompt_log_prob > max_likelihood:
			max_likelihood = prompt_log_prob
			library_idx = k
			
	return library_idx


def main():
	args, logger = get_config()
	evaluate(args, logger)


if __name__ == '__main__':
	main()
	
