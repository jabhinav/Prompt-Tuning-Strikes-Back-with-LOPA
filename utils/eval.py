import copy
import json
import os
import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def save_predictions_mbxp_format(
		args,
		output: Dict[str, List[str]],
		lang='python',
		d_type='MBPP',
		lib_size=None
):
	"""
	Save the predictions in the format required by the MBXP evaluation script.
	:param args:
	:param output:
	:param lang:
	:param d_type:
	:return:
	"""
	
	if args.do_peft and lib_size is not None:
		# Save each library's predictions in a separate file
		for k in range(lib_size):
			with open(os.path.join(args.log_dir, f'mbxp_solutions_lib_{k}.json'), 'w') as file:
				for problem in output:
					for response in output[problem][f'lib_{k}']:
						result_dict: dict = {
							"task_id": problem,
							"language": lang,
							"completion": response,
							"data_type": d_type
						}
						file.write(json.dumps(result_dict) + '\n')
			
			logger.info(f"Saved predictions for library {k} in the format required by the MBXP evaluation script")
	
	# Flatten all the predictions in a single file
	with open(os.path.join(args.log_dir, f'mbxp_solutions.json'), 'w') as file:
		for problem in output:
			if args.do_peft and lib_size is not None:
				for k in range(lib_size):
					for response in output[problem][f'lib_{k}']:
						result_dict: dict = {
							"task_id": problem,
							"language": lang,
							"completion": response,
							"data_type": d_type
						}
						file.write(json.dumps(result_dict) + '\n')
			else:
				for response in output[problem]:
					result_dict: dict = {
						"task_id": problem,
						"language": lang,
						"completion": response,
						"data_type": d_type
					}
					file.write(json.dumps(result_dict) + '\n')
	
	logger.info(f"Saved all predictions in a single file in the format required by the MBXP evaluation script")


def save_best_lib_predictions_mbxp_format(
		args,
		output: Dict[str, Dict[str, List[str]]],
		lib_mapping: Dict[str, int],
		lang='python',
		d_type='MBPP'
):
	"""
	Save the predictions in the format required by the MBXP evaluation script.
	:param args:
	:param output:
	:param lib_mapping:
	:param lang:
	:param d_type:
	:return:
	"""
	
	# Flatten all the predictions in a single file
	with open(os.path.join(args.log_dir, f'mbxp_solutions_best_lib.json'), 'w') as file:
		for problem in output:
			k = lib_mapping[problem]
			for response in output[problem][f'lib_{k}']:
				result_dict: dict = {
					"task_id": problem,
					"language": lang,
					"completion": response,
					"library": f"lib_{k}",
					"data_type": d_type
				}
				file.write(json.dumps(result_dict) + '\n')
	
	logger.info(f"Saved best lib predictions in a single file in the format required by the MBXP evaluation script")
	
	
def decode_mbpp_predictions(args, gen_token_dict, tokenizer, dataset) -> Tuple[List[List[Optional[str]]], List[List[Optional[str]]]]:
	
	code_gens: List[List[Optional[str]]] = [[] for _ in range(len(dataset))]
	
	# Remove the following pre-defined prefix from the generated tokens
	prefix = ''
	for task_idx, list_of_preds in gen_token_dict.items():
		for _pred in list_of_preds:
			
			# Remove the prompt from the generated code. This works when for the max_prompt_length, s only has prompt
			_pred = _pred[args.max_prompt_length:]
			
			# Remove the bos token if it's present
			if _pred[0] == tokenizer.bos_token_id:
				_pred = _pred[1:]
			
			# Treat eos token as a regular stop word not removing it from the output
			# If it's removed it may have the effect of removing it in the middle of a
			# longer generation in case a batch size > 1 is used, which will result in
			# a wrong generation as it won't be used for splitting lateron
			gen_code = tokenizer.decode(
				_pred, skip_special_tokens=False, clean_up_tokenization_spaces=False
			)
			try:
				# some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
				tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
				if gen_code.startswith(f"{tokenizer_prefix}"):
					gen_code = gen_code[len(tokenizer_prefix):].lstrip()
			except:
				pass
			
			# Split the response at the first occurrence of the end of text token.
			gen_code = gen_code.split(tokenizer.eos_token)[0]
			
			gen_code = gen_code[len(prefix):]
			code_gens[task_idx].append(gen_code)

	
	return code_gens, code_gens


def decode_cruxeval_predictions(gen_token_dict, tokenizer, dataset) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
	code_gens = defaultdict(list)
	code_gens_raw = defaultdict(list)
	
	# Remove the following pre-defined prefix from the generated tokens
	for task_idx, list_of_preds in gen_token_dict.items():
		for _pred in list_of_preds:
			
			# For empty predictions
			if len(_pred) == 0:
				code_gens[task_idx].append("")
				code_gens_raw[task_idx].append("")
				continue
			
			gen_text = tokenizer.decode(
				_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False
			)  # Remove special tokens else postprocess fn will fail
			try:
				# some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
				tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
				if gen_text.startswith(f"{tokenizer_prefix}"):
					gen_text = gen_text[len(tokenizer_prefix):].lstrip()
			except:
				pass

			# Post-process the generated text
			try:
				processed_text = dataset.task.postprocess_generation(gen_text, task_idx)
			except:
				print(f"Error in postprocessing for {task_idx}")
				processed_text = ""
			
			code_gens_raw[dataset.idx_to_id[task_idx]].append(gen_text)
			code_gens[dataset.idx_to_id[task_idx]].append(processed_text)
	
	return code_gens, code_gens_raw


def stardard_tokenize(sent):
	sent = ' '.join(re.split('(\W)', sent))
	sent = sent.split()
	sent = ' '.join(sent)
	return sent


def post_process(sent, is_tokenize, is_lower):
	if is_lower:
		sent = sent.lower()
	if is_tokenize:
		sent = stardard_tokenize(sent)
	return sent


def decode_nlg_predictions(task_name, gen_token_dict, tokenizer, dataset, is_tokenize=False, is_lower=False):

	
	refer_dict = {}

	# Remove the following pre-defined prefix from the generated tokens
	for task_idx, list_of_preds in gen_token_dict.items():
		
		# Due to number of test samples not a multiple of num processes, we may have some extra predictions when running in
		# parallel. For e2e, we will only consider the first prediction for each example.
		try:
			assert len(list_of_preds) == 1
		except AssertionError as e:
			print("E2E should have only one prediction per example. Found: ", list_of_preds)
			list_of_preds = list_of_preds[:1]
		
		# Let's get its references
		context, completion = dataset.get_sample(task_idx)
		if context not in refer_dict:
			refer_dict[context] = {}
			refer_dict[context]['references'] = []
		refer_dict[context]['references'].append(completion.split('<|endoftext|>')[0].split('\n\n')[0].strip())
		
		if task_name in ['nlg_webnlg', 'nlg_dart']:
			# Store the cate indicator
			refer_dict[context]['cate'] = dataset.category_seen(task_idx)
		
		for _pred in list_of_preds:
				
			# Remove the prompt from the prediction.
			input_len = dataset.input_lens[task_idx]
			_pred = _pred[input_len:]
			
			gen_text = tokenizer.decode(
				_pred, skip_special_tokens=False, clean_up_tokenization_spaces=False
			)
			
			refer_dict[context]['generated_text'] = gen_text
			
			# We will only consider one prediction in cases where,
			# > the model predicts multiple completions for given context, we save the last one
			# > data has multiple copies of the same context (w different references). Since we are doing greedy, completion will always be the same
			refer_dict[context]['hypothesis'] = gen_text.split('<|endoftext|>')[0].split('\n\n')[0].strip()  # This should give the prediction before EOS
			
			# Sanity-check: Remove any \n from the prediction (We always want single-line prediction, sometimes in PT it's multi-line)
			refer_dict[context]['hypothesis'] = refer_dict[context]['hypothesis'].replace('\n', ' ')
			
	# Post-process
	for context in refer_dict:
		refer_dict[context]['hypothesis'] = post_process(refer_dict[context]['hypothesis'], is_tokenize, is_lower)
		refer_dict[context]['references'] = [post_process(ref, is_tokenize, is_lower) for ref in refer_dict[context]['references']]
	
	return refer_dict, refer_dict


def decode_predictions(args, gen_token_dict, tokenizer, dataset):
	if 'cruxeval' in args.task_name:
		return decode_cruxeval_predictions(gen_token_dict, tokenizer, dataset)
	elif args.task_name == 'mbpp':
		return decode_mbpp_predictions(args, gen_token_dict, tokenizer, dataset)
	elif args.task_name in ['nlg_e2e', 'nlg_webnlg', 'nlg_dart']:
		if args.task_name == 'nlg_e2e':
			return decode_nlg_predictions(args.task_name, gen_token_dict, tokenizer, dataset)
		else:
			return decode_nlg_predictions(args.task_name, gen_token_dict, tokenizer, dataset, True, True)
	else:
		raise NotImplementedError(f"Decode Predictions for Task {args.task_name} not implemented yet!")