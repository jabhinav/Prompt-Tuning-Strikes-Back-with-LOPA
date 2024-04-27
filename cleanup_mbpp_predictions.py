import copy
import os
import json
from utils.prog import cleanup_return_statement, find_end_of_expression
import logging


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


def post_process_deepseek_predictions():
	# Post-process deepseek-coder-1.3b-base peft-tuned predictions. Remove BOS token
	
	dir = 'logging/deepseek-coder-7b-base_raw_cvae/results'
	
	logging.basicConfig(filename=os.path.join(dir, 'post_processing_logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	file = os.path.join(dir, 'mbxp_solutions.json')
	
	data = []
	with open(file, 'r') as f:
		for line in f:
			data.append(json.loads(line))
	
	experimental_data = copy.deepcopy(data)
	
	# Post-process: remove <｜begin▁of▁sentence｜> from the beginning of each completion
	for i in range(len(data)):
		task_id = data[i]['task_id'] if 'task_id' in data[i] else i
		
		# # Debug
		# if task_id == 'MBPP/523':
		# 	print("Debug")
		
		_prediction = data[i]['completion']
		
		# Filter 1: Remove <｜begin▁of▁sentence｜> from the beginning of each completion
		edited_completion = _prediction.replace('<｜begin▁of▁sentence｜>', '')
		
		# Filter 2: stop at the first occurrence of any stop token
		edited_completion = _stop_at_stop_token(edited_completion, MBPP_STOP_WORDS)
		data[i]['completion'] = edited_completion
		
		# Filter 3: [Unstable] Cleanup the return statement
		logger.info(f"\n\n============================ {task_id} =====================================")
		logger.info(f"Before return cleanup:\n\n{edited_completion}\n")
		edited_completion = cleanup_return_statement(edited_completion)
		logger.info(f"After return cleanup:\n\n{edited_completion}\n")
		
		experimental_data[i]['completion'] = edited_completion
		
		if data[i]['completion'].strip() != experimental_data[i]['completion'].strip():
			print("d")
		
	new_file = os.path.join(dir, 'mbxp_solutions_post_processed_stage1.json')
	with open(new_file, 'w') as f:
		for item in data:
			f.write(json.dumps(item) + '\n')
			
			
	new_file = os.path.join(dir, 'mbxp_solutions_post_processed_stage2.json')
	with open(new_file, 'w') as f:
		for item in experimental_data:
			f.write(json.dumps(item) + '\n')
			

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
	# post_process_deepseek_predictions()
	post_process_stop_at_stop_token()

