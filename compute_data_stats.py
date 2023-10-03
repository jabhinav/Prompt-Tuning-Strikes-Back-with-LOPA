import os
from tqdm import tqdm

from utils.model import load_tokenizer, get_huggingface_path
from typing import List

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
