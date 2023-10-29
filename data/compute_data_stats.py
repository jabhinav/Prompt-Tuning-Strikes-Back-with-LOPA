import os
import json
from tqdm import tqdm

from utils.xformer import load_tokenizer, get_huggingface_path
from utils.data import MBPP_Dataset as CustomDataset
from typing import List


def compute_stats():
	# Get the tokenizer
	model_type = "codegen-350M"
	huggingface_path = get_huggingface_path(model_type)
	tokenizer = load_tokenizer(model_type, huggingface_path)
	
	# Read data
	# Read (processed) python files from the given path
	path_to_data = './sample_codes_processed'
	files = os.listdir(path_to_data)
	files = [f for f in files if f.endswith('.py')]
	
	token_lens: List[int] = []
	for f in tqdm(files, desc='Reading files', total=len(files)):
		with open(os.path.join(path_to_data, f), 'r') as file:
			prog_instructions: List[str] = file.readlines()
		
		program = '\n'.join(prog_instructions)
		program_tokens = tokenizer.tokenize(program)
		token_lens.append(len(program_tokens))
		
	# Print Mean, Max, Min
	print("Mean: {}, Max: {}, Min: {}".format(sum(token_lens) / len(token_lens), max(token_lens), min(token_lens)))
	
	
def create_split():
	# Get the tokenizer
	model_type = "codegen-350M"
	huggingface_path = get_huggingface_path(model_type)
	tokenizer = load_tokenizer(model_type, huggingface_path)
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data="./data/MBPP/mbpp_release_v1.jsonl",
		tokenizer=tokenizer,
		max_prompt_length=325,
		max_length=325+256,
		sample_problems=None,
		mode='test'
	)
	
	# Get task ids
	test_tasks = dataset.get_ids()
	
	# Create a separate jsonl file for test tasks
	with open('./data/MBPP/mbpp_release_v1.jsonl', 'r') as f:
		data = f.readlines()
	
	data = [json.loads(d) for d in data]
	data = [d for d in data if d['task_id'] in test_tasks]
	
	with open('./data/MBPP/mbpp_test_release_v1.jsonl', 'w') as f:
		for d in data:
			f.write(json.dumps(d) + '\n')
			

if __name__ == '__main__':
	# compute_stats()
	create_split()