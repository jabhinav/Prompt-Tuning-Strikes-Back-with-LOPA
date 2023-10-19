import glob
import json
import os
import random
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.reindent import reindent_code


class LibraryBaseDataset(Dataset):
	def __init__(self, path_to_data: str, tokenizer, max_prompt_length, max_response_length):
		self.path_to_data = path_to_data
		self.tokenizer = tokenizer
		self.max_prompt_length = max_prompt_length
		self.max_response_length = max_response_length
		self.max_length = max_prompt_length + max_response_length
		
		self.data = self.read_data()
		self.ids: List[str] = list(self.data.keys())
	
	def read_data(self):
		raise NotImplementedError
	
	def sample(self, idx: int):
		raise NotImplementedError
	
	def process(self, src_input_ids, trg_label_ids):
		
		if len(src_input_ids) < self.max_length:
			# Pad input with eos token
			new_input_ids = [self.tokenizer.eos_token_id] * self.max_length
			new_input_ids[:len(src_input_ids)] = src_input_ids
			src_input_ids = new_input_ids
			
			# Pad label with -100
			new_label_ids = [-100] * self.max_length
			new_label_ids[:len(trg_label_ids)] = trg_label_ids
			trg_label_ids = new_label_ids
		
		# Convert to tensors
		src_input_ids = torch.LongTensor(src_input_ids)
		trg_label_ids = torch.LongTensor(trg_label_ids)
		
		src_mask = src_input_ids.ne(self.tokenizer.eos_token_id)  # mask out padding
		trg_mask = trg_label_ids.ne(-100)  # mask out padding
		
		return src_input_ids, src_mask, trg_label_ids, trg_mask
	
	def __getitem__(self, idx):
		return self.sample(idx)
	
	def __len__(self):
		return len(self.data)
	
	def __repr__(self):
		return f"Dataset({self.path_to_data})"
	
	def __str__(self):
		return f"Dataset({self.path_to_data})"
	
	def __iter__(self):
		return iter(self.data)
	
	def __contains__(self, item):
		return item in self.data
	
	def __add__(self, other):
		return self.data + other.data


class LibrarySampleDataset(LibraryBaseDataset):
	
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int,
			max_response_length: int,
			sample_problems: Union[int, None] = None
	):
		
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_response_length)
	
	def read_data(self, read_labels: bool = False):
		
		"""
		Each file follows the convention k<cluster_id>_<specific_method_name>_<library>.py
		
		:return:
		"""
		
		# Read (processed) python files from the given path
		files = os.listdir(self.path_to_data)
		files = [f for f in files if f.endswith('.py')]
		
		f_ids: List[str] = []
		data: Dict[str, str] = {}
		for f in tqdm(files, desc='Reading files', total=len(files)):
			with open(os.path.join(self.path_to_data, f), 'r') as file:
				prog_instructions: List[str] = file.readlines()
			
			program = '\n'.join(prog_instructions)
			# label = f.split('_')[-1].split('.')[0] if read_labels else None
			data[f] = program
		
		return data
	
	def sample(self, idx: int):
		program = self.data[self.ids[idx]]
		
		program_tokens = self.tokenizer.tokenize(program)
		# label_tokens = self.tokenizer.tokenize(label) if label is not None else None
		
		# Truncate the program + label tokens if it exceeds the max length (from left)
		max_length = self.max_length - 1  # for start token
		if len(program_tokens) > max_length:
			program_tokens = program_tokens[-max_length:]
		
		program_token_ids = self.tokenizer.convert_tokens_to_ids(program_tokens)
		
		src_input_ids = [self.tokenizer.bos_token_id] + program_token_ids
		trg_label_ids = program_token_ids + [self.tokenizer.eos_token_id]
		
		return self.process(src_input_ids, trg_label_ids)


def generate_apps_prompt(problem, tokenizer=None, peek_frac=0.0):
	prob_path = os.path.join(problem)
	test_case_path = os.path.join(prob_path, "input_output.json")
	question_fname = os.path.join(prob_path, "question.txt")
	starter_code = os.path.join(prob_path, "starter_code.py")
	sols_fname = os.path.join(prob_path, "solutions.json")
	
	if not os.path.exists(starter_code):
		starter_code = None
	
	# Read the Question
	_input = "\nQUESTION:\n"
	with open(question_fname, "r") as f:
		data = f.readlines()
		data = "".join(data)
	_input += data
	
	# If the starter code is provided, append it to the question
	if starter_code is not None:
		with open(starter_code, "r") as f:
			data = f.readlines()
			data = "".join(data)
			data = "\n" + data
		# print("Starter code is provided at {}: {}".format(starter_code, data))
		_input += data
	
	# Decide on the format for data - based on starter code.
	# If available, use the call-based format else use standard format
	if os.path.exists(test_case_path):
		with open(test_case_path, "r") as f:
			data = json.load(f)
		
		if not data.get("fn_name"):
			_input += "\nUse Standard Input format"
		else:
			_input += "\nUse Call-Based format"
	
	elif starter_code is not None and os.path.exists(starter_code):
		_input += "\nUse Call-Based format"
	else:
		_input += "\nUse Standard Input format"
	
	_input += "\nANSWER:\n"
	
	if peek_frac > 0.0 and tokenizer is not None:
		if os.path.exists(sols_fname):
			with open(sols_fname, 'r') as f:
				solns = json.load(f)
			
			sample_sol = random.choice(solns)
			rand_sol = reindent_code(sample_sol)
			# Do not provide max_length here since we want to see the full code and make a portion of it visible
			rand_sol = tokenizer.encode(rand_sol, verbose=False)
			tokens_taken = int(peek_frac * len(rand_sol))
			rand_sol = rand_sol[:tokens_taken]
			# Skip special tokens since we do not want them to be visible
			partial_prog = tokenizer.decode(rand_sol, skip_special_tokens=True)
			_input += partial_prog
	
	return _input


class APPSBaseDataset(LibraryBaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_response_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None
	):
		
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.mode = mode
		path_to_data = os.path.join(path_to_data, 'train')
		
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_response_length)
	
	def read_data(self):
		"""
		Assume self.dataroot is set to folderName/data
		"""
		skipped_problems = []
		data = {}  # Mapping from question_fname to list of samples
		
		original_problems = glob.glob(self.path_to_data + '/*')
		problems = sorted(original_problems)
		
		if self.sample_problems is not None:
			problems = problems[:self.sample_problems]  # Keep it fixed for reproducibility to first 'sample' problems
		
		for prob_index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems),
										desc="Reading APPS examples from {}: ".format(self.path_to_data)):
			problem_id = problem.split('/')[-1]
			prob_path = os.path.join(problem)
			sols_fname = os.path.join(prob_path, "solutions.json")
			
			# Solutions are required when mode is train or val
			if self.mode in ['train', 'dev'] and not os.path.isfile(sols_fname):
				skipped_problems.append(prob_path)
				continue
			
			q_str = generate_apps_prompt(problem)
			
			if self.mode in ['train', 'dev']:
				# Read all the solutions
				with open(sols_fname, 'r') as f:
					sols_str_list = json.load(f)
				
				for a_idx, a_str in enumerate(sols_str_list):
					f_idx = '{}:{}'.format(problem_id, a_idx)
					a_str = reindent_code(a_str)
					data[f_idx] = (q_str, a_str)
			else:
				# In APPS, there are multiple solutions for each problem! For now, we won't store them
				f_idx = '{}'.format(problem_id)
				data[f_idx] = (q_str, '')
		
		print(f"Skipped {len(skipped_problems)} problems! Mode is {self.mode}")
		return data
	
	def sample(self, idx: int):
		
		q_str, a_str = self.data[self.ids[idx]]
		question_token_ids = self.tokenizer.encode(q_str, verbose=False)
		question_token_ids = question_token_ids[-self.max_prompt_length:]  # Truncate the prompt from left
		
		if self.mode == 'test':
			# No need to pad for test and add the solution
			question_token_ids = torch.LongTensor(question_token_ids)
			mask = question_token_ids.ne(self.tokenizer.eos_token_id)
			return question_token_ids, mask
		
		else:
			answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
			answer_token_ids.append(self.tokenizer.eos_token_id)
			
			src_input_ids = []
			trg_label_ids = []
			src_input_ids.extend(question_token_ids)
			src_input_ids.extend(answer_token_ids)
			trg_label_ids.extend([-100] * len(question_token_ids))
			trg_label_ids.extend(answer_token_ids)
			
			# CUT OFF THE EXCESS
			if len(src_input_ids) > self.max_length:
				# Truncate i/p, label from right (this will auto. truncate the response)
				src_input_ids = src_input_ids[self.max_length:]
				trg_label_ids = trg_label_ids[self.max_length:]
				
			return self.process(src_input_ids, trg_label_ids)
