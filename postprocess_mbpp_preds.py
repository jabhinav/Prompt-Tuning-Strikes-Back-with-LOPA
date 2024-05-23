import copy
import os
import json
from utils.prog import cleanup_return_statement, find_end_of_first_valid_python_expression
import logging
import argparse
import autopep8


MBPP_STOP_WORDS = ["\ndef", "\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```", '<｜end▁of▁sentence｜>']



def _stop_at_stop_token(decoded_string, stop_tokens):
	"""
	Produces the prefix of decoded_string that ends at the first occurrence of
	a stop_token.
	WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
	itself.
	"""
	min_stop_index = len(decoded_string)
	for stop_token in stop_tokens:
		stop_index = decoded_string.find(stop_token)
		if stop_index != -1 and stop_index < min_stop_index:
			min_stop_index = stop_index
	return decoded_string[:min_stop_index]


# G.T. data path
test_file = './mxeval/mbpp_test_release_v1.jsonl'
with open(test_file, 'r') as f:
	data = f.readlines()
	
data = [json.loads(d) for d in data]
# Convert to a dictionary
data = {item['task_id']: item['prompt'] for item in data}


def correct_indent(prompt, completion):
	"""
	Correct the indent of the completion using the ground truth indent
	:param prompt:
	:param completion:
	:return:
	"""
	# First line is a def statement, get the indent from the next line
	assert prompt.strip().startswith('def'), f"Prompt does not start with 'def': {prompt}"
	indent = len(prompt.split('\n')[1]) - len(prompt.split('\n')[1].lstrip())
	running_indent_exp = prompt.split('\n')[1][:indent]
	
	# Add dummy function header to the completion
	dummy_completion = 'def dummy_function():\n' + completion
	dummy_completion = autopep8.fix_code(dummy_completion)  # This will always add spaces for indent
	# Remove the dummy function header to get the correct completion
	dummy_completion = dummy_completion.split('\n')[1:]
	correct_completion = '\n'.join(dummy_completion)
	
	# If the prompt uses tabs, replace the spaces in completion with tabs otherwise code will not run
	if '\t' in running_indent_exp:
		correct_completion = correct_completion.replace('    ', '\t')
	# If only white spaces -> then replace to use the same indent exp as the prompt
	else:
		correct_completion = correct_completion.replace('    ', running_indent_exp)
	
	return correct_completion


def special_meta_llama_3_8b_cleanup(decoded_string, task_id):
	"""
	Sometimes the Meta-Llama-3-8B model produces garbage instructions starting with ":" or "//" etc.
	It might also make mistakes in indenting the completion.
	:param decoded_string:
	:param task_id:
	:return:
	"""
	
	# Filter 4: [Meta-Llama3-8B specific] Remove garbage instructions starting with ":" or "//"
	decoded_string = decoded_string.split('\n')
	new_beta_prediction = []
	for line in decoded_string:
		if line.strip().startswith(':') or line.strip().startswith('//') or line.strip().startswith('.'):
			continue
		new_beta_prediction.append(line)
	decoded_string = '\n'.join(new_beta_prediction)
	
	# Correct the indent of the completion
	gt_prompt = data[task_id].strip()
	corrected_string = correct_indent(gt_prompt, decoded_string)
	
	return corrected_string


def post_process(args, db=False):
	assert os.path.exists(args.path), f"Path does not exist: {args.path}"
	
	if db:
		log_at = './logging/post_processing_logs.txt'
		logging.basicConfig(filename=log_at, filemode='w',
							format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
							datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
		logger = logging.getLogger(__name__)
	
	data = []
	with open(args.path) as f:
		for line in f:
			data.append(json.loads(line))
	
	experimental_data = copy.deepcopy(data)
	
	# Post-process: remove <｜begin▁of▁sentence｜> from the beginning of each completion
	for i in range(len(data)):
		task_id = data[i]['task_id'] if 'task_id' in data[i] else i
		
		if task_id == 'MBPP/700' and db:
			print('here')
		
		orig_prediction = data[i]['completion']
		
		# Filter 1: Remove <｜begin▁of▁sentence｜> from the beginning of each completion
		if '<｜begin▁of▁sentence｜>' in orig_prediction:
			_prediction = orig_prediction.replace('<｜begin▁of▁sentence｜>', '')
		else:
			_prediction = orig_prediction
		
		# Filter 2: stop at the first occurrence of any stop token
		_prediction = _stop_at_stop_token(_prediction, MBPP_STOP_WORDS)
		data[i]['completion'] = _prediction
		
		# # Filter 3: [Beta]
		if 'Meta-Llama-3-8B' in args.path or db:
			beta_prediction = special_meta_llama_3_8b_cleanup(_prediction, task_id)
		else:
			beta_prediction = cleanup_return_statement(_prediction)
		
		if db:
			logger.info(f"\n\n============================ {task_id} =====================================")
			if _prediction.strip() != beta_prediction.strip():
				# logger.info("Original prediction:\n\n" + orig_prediction + "\n")
				logger.info(f"After stopword removal:\n\n{_prediction}\n")
				logger.info(f"After return cleanup:\n\n{beta_prediction}\n")
		
		experimental_data[i]['completion'] = beta_prediction
	
	if not db:
		new_path = args.path.replace('mbxp_solutions.json', 'mbxp_solutions_post_processed_stage1.json')
		with open(new_path, 'w') as f:
			for item in data:
				f.write(json.dumps(item) + '\n')
	
	
		new_path = args.path.replace('mbxp_solutions.json', 'mbxp_solutions_post_processed_stage2.json')
		with open(new_path, 'w') as f:
			for item in experimental_data:
				f.write(json.dumps(item) + '\n')
				
		new_path = args.path.replace('mbxp_solutions.json', 'mbxp_solutions_post_processed.json')
		with open(new_path, 'w') as f:
			# Dump both the original and the experimental data
			for item1, item2 in zip(data, experimental_data):
				f.write(json.dumps(item1) + '\n')
				f.write(json.dumps(item2) + '\n')
			

def post_process_stop_at_stop_token():
	
	dir = 'logging/phi-2_results'
	file = os.path.join(dir, 'mbxp_solutions.json')
	
	data = []
	with open(file, 'r') as f:
		for line in f:
			data.append(json.loads(line))
	
	# Post-process: stop at the first occurrence of any stop token
	for i in range(len(data)):
		edited_completion = _stop_at_stop_token(data[i]['completion'], MBPP_STOP_WORDS)
		data[i]['completion'] = edited_completion
		
	new_file = os.path.join(dir, 'mbxp_solutions_post_processed.json')
	with open(new_file, 'w') as f:
		for item in data:
			f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default=None)
	_args = parser.parse_args()
	post_process(_args)
	# post_process_stop_at_stop_token()
	

