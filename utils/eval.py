import json
import os
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
	for sample, generated_tokens in gen_token_dict.items():
		for s in generated_tokens:
			
			# Remove the prompt from the generated code. This works when for the max_prompt_length, s only has prompt
			s = s[args.max_prompt_length:]
			
			# Remove the bos token if it's present
			if s[0] == tokenizer.bos_token_id:
				s = s[1:]
			
			# Treat eos token as a regular stop word not removing it from the output
			# If it's removed it may have the effect of removing it in the middle of a
			# longer generation in case a batch size > 1 is used, which will result in
			# a wrong generation as it won't be used for splitting lateron
			gen_code = tokenizer.decode(
				s, skip_special_tokens=False, clean_up_tokenization_spaces=False
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
			code_gens[sample].append(gen_code)

	
	return code_gens, code_gens


def decode_cruxeval_predictions(gen_token_dict, tokenizer, dataset) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
	code_gens = defaultdict(list)
	code_gens_raw = defaultdict(list)
	
	# Remove the following pre-defined prefix from the generated tokens
	for _idx, list_of_preds in gen_token_dict.items():
		for _pred in list_of_preds:
			
			# For empty predictions
			if len(_pred) == 0:
				code_gens[_idx].append("")
				code_gens_raw[_idx].append("")
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
				processed_text = dataset.task.postprocess_generation(gen_text, _idx)
			except:
				print(f"Error in postprocessing for {_idx}")
				processed_text = ""
			
			code_gens_raw[dataset.idx_to_id[_idx]].append(gen_text)
			code_gens[dataset.idx_to_id[_idx]].append(processed_text)
	
	return code_gens, code_gens_raw


def decode_predictions(args, gen_token_dict, tokenizer, dataset):
	if 'cruxeval' in args.task_name:
		return decode_cruxeval_predictions(gen_token_dict, tokenizer, dataset)
	elif args.task_name == 'mbpp':
		return decode_mbpp_predictions(args, gen_token_dict, tokenizer, dataset)
	else:
		raise NotImplementedError(f"Decode Predictions for Task {args.task_name} not implemented yet")