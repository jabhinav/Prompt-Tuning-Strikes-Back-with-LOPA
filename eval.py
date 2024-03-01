import json
import logging
from collections import OrderedDict
from typing import Dict, List

import torch
import os
import numpy as np
from tqdm import tqdm

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftMultiModel
from utils.model import get_response_log_probs_for_lib
from utils.custom import save_best_lib_predictions_mbxp_format, save_predictions_mbxp_format, log_dist
from utils.config import get_config
from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path
from utils.model import ClarificationCodeBERTPredictor


@torch.no_grad()
def get_clf_idx_using_prompt_ll(args, prompt, prompt_mask, tokenizer, model) -> int:
	"""
	Compute the library index using the log-likelihood of the prompt coming from the latent prompt of each library.
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
		prompt_log_prob = get_response_log_probs_for_lib(args, (prompt, prompt_mask, response, response_mask), tokenizer, model,
														 k)
		
		# Update the library index
		if prompt_log_prob > max_likelihood:
			max_likelihood = prompt_log_prob
			library_idx = k
	
	return library_idx


@torch.no_grad()
def get_clf_idx_using_NN(args, logger):
	"""
	Compute the library index using a NN classifier.
	:param args:
	:param logger:
	:return: library_idx: int
	"""
	
	# Load checkpoint
	if args.clf_predictor_path is None:
		logger.info("Prior checkpoint not specified.")
		return None
	
	if not os.path.exists(args.clf_predictor_path):
		logger.info("Prior checkpoint not found at: {}".format(args.clf_predictor_path))
		return None
	
	# test_prompt_embeddings = torch.load('./logging/test_prompt_embeddings.pkl')
	# input_dim = test_prompt_embeddings[list(test_prompt_embeddings.keys())[0]].size(-1)
	
	# Add BERT specific args
	args.bert_model_type = "codebert-base"
	args.bert_tokenizer_name = get_huggingface_path(args.bert_model_type)
	args.bert_model_name_or_path = get_huggingface_path(args.bert_model_type)
	args.bert_config_name = get_huggingface_path(args.bert_model_type)
	msg = f"Added model specific args for {args.bert_model_type}"
	log_dist(message=msg, level=logging.INFO, ranks=[0])
	
	model = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_libraries)
	
	loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
	loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
	
	# # [For backward compt.] Replace the keys prob_head.weight, prob_head.bias with out_proj.weight, out_proj.bias
	# loaded_state_dict['out_proj.weight'] = loaded_state_dict.pop('prob_head.weight')
	# loaded_state_dict['out_proj.bias'] = loaded_state_dict.pop('prob_head.bias')
	
	model.load_state_dict(loaded_state_dict, strict=True)
	del loaded_state_dict
	
	print("[INFO] Loaded the clarification index predictor from: {}".format(args.clf_predictor_path))
	
	# GPU-ize the model
	model.to(args.device)
	model.eval()
	
	def get_clf_idx(bert_inputs_ids, bert_mask):
		
		# When using BERT-like model as NN classifier
		logits = model(bert_inputs_ids, bert_mask)
		probabilities = torch.nn.functional.softmax(logits, dim=-1)
		probabilities = probabilities.detach().cpu()
		
		# # If argmax is the chosen library index
		# indices = torch.argmax(probabilities, dim=1)
		# pred_clf_idx = indices.detach().cpu().item()
		
		probabilities = probabilities.numpy()[0]
		# If argmax is the chosen library index
		pred_clf_idx = np.argmax(probabilities)
		# If sampling is the chosen library index
		# pred_clf_idx = np.random.choice(np.arange(args.num_libraries), p=probabilities)
		
		return pred_clf_idx
	
	return get_clf_idx


@torch.no_grad()
def evaluate(args, logger):
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test',
	
		# # Uncomment to use a finer split of the training data to evaluate
		# finer_split=0.50,
		# use_first_half=False
	)
	
	# # Leave this as is to only read prompt for any type of data
	dataset.mode = 'test'
	
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
	
	# Update the model's padding token id for open-ended generation
	if 't5' not in args.model_type and model.config.pad_token_id is None:
		model.config.pad_token_id = tokenizer.pad_token_id
	
	if args.do_peft:
		
		if not os.path.exists(args.load_adapter_from):
			logger.error("Please specify the correct path to load the model adapters from")
			raise ValueError("Please specify the correct path to load the model adapters from")
		
		# Get the config
		peft_config = PromptTuningConfig(
			task_type=TaskType.MULTI_CAUSAL_LM,
			# CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=args.num_virtual_tokens,
			# prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
			# tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
			num_init_clusters=args.num_libraries,  # My custom field
		)

		# Load the model adapters - in place
		model = PeftMultiModel.from_pretrained(
			model=model,
			model_id=args.load_adapter_from,  # Must be a directory containing the model files
			config=peft_config,
		)
		msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
		logger.info(msg)
		print(msg)
	
	# GPU-ize the model
	model.to(args.device)
	
	# Predict for each sample output by each library
	num_loops = int(args.num_return_sequences / args.num_return_sequences_per_iter)
	
	oracle_output: Dict[str, Dict[str, List[str]]] = {}
	lib_mapping:  Dict[str, int] = {}
	get_clf_idx = get_clf_idx_using_NN(args, logger)  # Define the function to get the library index
	model.eval()
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		# input_embeds, input_ids, attention_mask = sample
		bert_inputs_ids, bert_mask, input_ids, attention_mask = sample
		
		# ################################################ PEFT ################################################# #
		if args.do_peft:
		
			chosen_clf_idx = None
			if args.do_peft and get_clf_idx is not None:
				# chosen_lib_idx = get_clf_idx_using_prompt_ll(args, input_ids, attention_mask, tokenizer, model)
				# chosen_lib_idx = get_clf_idx(input_embeds)  # For MLP as NN classifier
				chosen_clf_idx = get_clf_idx(bert_inputs_ids, bert_mask)  # For BERT-like model as NN classifier
				
			all_library_predictions: Dict[str, List[str]] = OrderedDict()
			for k in range(args.num_libraries):
				
				# Set the library index
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
			
			if chosen_clf_idx is not None:
				lib_mapping[dataset.ids[index]] = int(chosen_clf_idx)
				
		# ############################################### No PEFT ############################################### #
		else:
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
				
				
			oracle_output[dataset.ids[index]] = all_responses
			
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(oracle_output, f, indent=4)
	
	# Save the output for the best chosen libraries
	if lib_mapping:
		save_best_lib_predictions_mbxp_format(args, oracle_output, lib_mapping, lang='python', d_type='MBPP')
	
	save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP', lib_size=args.num_libraries)


def main():
	args, logger = get_config()
	
	# # Debug
	# args.do_peft = 1
	# args.load_base_from_path = './logging/Baseline_0.50/output/pytorch_model.bin'
	# args.load_adapter_from = './logging/PEFT_Oracle_0.50_0.50_20ep/PromptTuningMultiModel'
	
	evaluate(args, logger)


if __name__ == '__main__':
	main()
	
