import glob
import json
import os
import random
import logging
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.reindent import reindent_code
from utils.xformer import load_tokenizer, get_huggingface_path


logger = logging.getLogger(__name__)


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


class LibraryBaseDataset(Dataset):
	def __init__(self, path_to_data: str, tokenizer, max_prompt_length, max_length, mode: str, return_dict: bool = False):
		self.path_to_data = path_to_data
		self.tokenizer = tokenizer
		self.max_prompt_length = max_prompt_length
		self.max_length = max_length
		self.mode = mode
		self.return_dict = return_dict
		
		# Read Data
		self.data = self.read_data()
		self.ids: List[str] = list(self.data.keys())
	
	def read_data(self):
		raise NotImplementedError
	
	def sample(self, idx: int):
		
		src_input_ids, trg_label_ids = [], []
		q_str, a_str = self.data[self.ids[idx]]
		question_token_ids = self.tokenizer.encode(q_str, verbose=False)
		question_token_ids = question_token_ids[-self.max_prompt_length:]  # Truncate the prompt from left
		src_input_ids.extend(question_token_ids)
		
		if self.mode == 'test':
			
			# Pad from left: Allows batched generation
			if len(src_input_ids) < self.max_prompt_length:
				new_input_ids = [self.tokenizer.eos_token_id] * self.max_prompt_length
				new_input_ids[-len(src_input_ids):] = src_input_ids
				src_input_ids = new_input_ids
				
			src_input_ids = torch.LongTensor(src_input_ids)
			mask = src_input_ids.ne(self.tokenizer.eos_token_id)
			return src_input_ids, mask
		
		else:
			answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
			answer_token_ids.append(self.tokenizer.eos_token_id)
			
			src_input_ids.extend(answer_token_ids)
			
			# # Want to generate prompt as part of the response
			# trg_label_ids.extend(question_token_ids)
			
			# # Want to generate response only
			trg_label_ids.extend([-100] * len(question_token_ids))
			trg_label_ids.extend(answer_token_ids)
			
			# Cut off the excess
			if len(src_input_ids) > self.max_length:
				# Truncate i/p, label from right (this will auto. truncate only the response)
				src_input_ids = src_input_ids[:self.max_length]
				trg_label_ids = trg_label_ids[:self.max_length]
			
			# # Print the shapes
			# print(f"[Debug] src_input_ids: {len(src_input_ids)}")
			# print(f"[Debug] trg_label_ids: {len(trg_label_ids)}")
			
			return self.process(src_input_ids, trg_label_ids)
	
	def process(self, src_input_ids, trg_label_ids):
		
		if len(src_input_ids) < self.max_length:
			# Pad input [prompt+response] with eos token from right
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
		
		if self.return_dict:
			return {
				"input_ids": src_input_ids,
				"attention_mask": src_mask,
				"labels": trg_label_ids
			}
		else:
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
	
	def get_ids(self):
		return self.ids


class MBPP_Dataset(LibraryBaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None,
			split: float = 0.5,
			finer_split: float = 1.0,
			use_first_half: bool = True,
	):
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.split = split
		
		# Use the first <finer_split> fraction of the data as training-1 and the rest as training-2
		self.finer_split = finer_split
		self.use_first_half = use_first_half
		
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_length, mode)
	
	def read_data(self):
		with open(self.path_to_data, 'r') as f:
			data = f.readlines()
		
		data = [json.loads(d) for d in data]
		
		# Divide into train-test
		if self.mode == 'train':
			data = data[:int(self.split * len(data))]
			
			# Divide into training-1 and training-2
			if self.finer_split < 1.0:
				training_1 = data[:int(self.finer_split * len(data))]
				training_2 = data[int(self.finer_split * len(data)):]
				data = training_1 if self.use_first_half else training_2
			
		elif self.mode == 'test':
			data = data[int(self.split * len(data)):]
		
		partitioned_data = {}
		for problem in tqdm(data, ncols=0, total=len(data),
							desc="Reading MBPP examples from {} (mode = {}): ".format(self.path_to_data,
																					  self.mode)):
			f_id = problem['task_id']
			q_str = problem['prompt']
			a_str = problem['canonical_solution']
			partitioned_data[f_id] = (q_str, a_str)
		
		return partitioned_data
	

class MBPP_Dataset_w_CodeBERT(LibraryBaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None,
			split: float = 0.5,
			finer_split: float = 1.0,
			use_first_half: bool = True,
	):
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.split = split
		
		# Use the first <finer_split> fraction of the data as training-1 and the rest as training-2
		self.finer_split = finer_split
		self.use_first_half = use_first_half
		
		# Load BERT tokenizer
		bert_model_type = 'codebert-base'
		logger.info(f"Loading tokenizer from {get_huggingface_path(bert_model_type)}")
		self.bert_tokenizer = load_tokenizer(bert_model_type, get_huggingface_path(bert_model_type))
		
		
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_length, mode)
	
	def read_data(self):
		
		# Read the MBPP data
		with open(self.path_to_data, 'r') as f:
			data = f.readlines()
		
		data = [json.loads(d) for d in data]
		
		# Divide into train-test
		if self.mode == 'train':
			data = data[:int(self.split * len(data))]
			
			# Divide into training-1 and training-2
			if self.finer_split < 1.0:
				training_1 = data[:int(self.finer_split * len(data))]
				training_2 = data[int(self.finer_split * len(data)):]
				data = training_1 if self.use_first_half else training_2
				
		elif self.mode == 'test':
			data = data[int(self.split * len(data)):]
		
		partitioned_data = {}
		for problem in tqdm(data, ncols=0, total=len(data),
							desc="Reading MBPP examples from {} (mode = {}): ".format(self.path_to_data,
																					  self.mode)):
			f_id = problem['task_id']
			q_str = problem['prompt']
			a_str = problem['canonical_solution']
			partitioned_data[f_id] = (q_str, a_str)
		
		return partitioned_data
	
	def get_bert_ip(self, idx):
		q_str, _ = self.data[self.ids[idx]]
		src_tokens = self.bert_tokenizer.tokenize(q_str)
		
		max_bert_length = self.max_prompt_length - 2  # [CLS] and [SEP]
		src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
		src_tokens = [self.bert_tokenizer.cls_token] + src_tokens + [self.bert_tokenizer.sep_token]
		src_token_ids = self.bert_tokenizer.convert_tokens_to_ids(src_tokens)
		
		# Pad from right
		pad_length = self.max_prompt_length - len(src_token_ids)
		src_token_ids = src_token_ids + [self.bert_tokenizer.pad_token_id] * pad_length
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.bert_tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask
		

	def sample(self, idx: int):
		"""
		Overriding the sample method to return the input for CodeBERT as well.
		:param idx:
		:return:
		"""
		
		_id = self.ids[idx]
		
		# Get the input for CodeBERT
		bert_input_ids, bert_mask = self.get_bert_ip(idx)
		
		if self.mode == 'train':
			src_input_ids, src_mask, trg_label_ids, trg_mask = super().sample(idx)
			return bert_input_ids, bert_mask, src_input_ids, src_mask, trg_label_ids, trg_mask
		
		else:
			src_input_ids, src_mask = super().sample(idx)
			return bert_input_ids, bert_mask, src_input_ids, src_mask


class MBPP_Dataset_only_CodeBERT(Dataset):
	def __init__(
			self,
			num_classes: int,
			path_to_data: str,
			path_to_labels: str = None,
			tokenizer: Any = None,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None,
			split: float = 0.5,
			finer_split: float = 1.0,
			use_first_half: bool = True,
	):
		self.num_classes = num_classes
		
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.split = split
		
		# Use the first <finer_split> fraction of the data as training-1 and the rest as training-2
		self.finer_split = finer_split
		self.use_first_half = use_first_half
		
		self.path_to_data = path_to_data
		self.path_to_labels = path_to_labels
		self.tokenizer = tokenizer
		self.max_prompt_length = max_prompt_length
		self.max_length = max_length
		self.mode = mode
		
		# Read Data
		self.data = self.read_data()
		self.labels = self.read_labels()
		self.cls_weights = self.get_cls_weights()
		self.ids: List[str] = list(self.data.keys())
		
	def read_data(self):
		
		# Read the MBPP data
		with open(self.path_to_data, 'r') as f:
			data = f.readlines()
		
		data = [json.loads(d) for d in data]
		
		# Divide into train-test
		if self.mode == 'train':
			data = data[:int(self.split * len(data))]
			
			# Divide into training-1 and training-2
			if self.finer_split < 1.0:
				training_1 = data[:int(self.finer_split * len(data))]
				training_2 = data[int(self.finer_split * len(data)):]
				data = training_1 if self.use_first_half else training_2
			
		elif self.mode == 'test':
			data = data[int(self.split * len(data)):]
		
		partitioned_data = {}
		for problem in tqdm(data, ncols=0, total=len(data),
							desc="Reading MBPP examples from {} (mode = {}): ".format(self.path_to_data,
																					  self.mode)):
			f_id = problem['task_id']
			q_str = problem['prompt']
			a_str = problem['canonical_solution']
			partitioned_data[f_id] = (q_str, a_str)
		
		return partitioned_data
	
	def read_labels(self):
		if self.path_to_labels is None:
			logger.info(f"Label path not provided for {self.mode}")
			return {}
		
		if os.path.exists(self.path_to_labels):
			with open(self.path_to_labels, 'r') as f:
				labels = json.load(f)
			return labels
		else:
			logger.info(f"Labels not found for {self.mode} at {self.path_to_labels}")
			return {}
		
	def get_cls_weights(self) -> torch.FloatTensor:
		
		if not self.labels:
			cls_weights = [1.0 for _ in range(self.num_classes)]
			cls_weights = torch.FloatTensor(cls_weights)
			return cls_weights
		
		cls_weights = [0.0 for _ in range(self.num_classes)]
		for _id, possible_labels in self.labels.items():
			for label in possible_labels:
				cls_weights[label] += 1.0
		
		cls_weights = [1.0 / w if w > 0 else 0.0 for w in cls_weights]
		cls_weights = torch.FloatTensor(cls_weights)
		
		return cls_weights
	
	def get_bert_ip(self, idx):
		q_str, _ = self.data[self.ids[idx]]
		src_tokens = self.tokenizer.tokenize(q_str)
		
		max_bert_length = self.max_prompt_length - 2  # [CLS] and [SEP]
		src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
		src_tokens = [self.tokenizer.cls_token] + src_tokens + [self.tokenizer.sep_token]
		src_token_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
		
		# Pad from right
		pad_length = self.max_prompt_length - len(src_token_ids)
		src_token_ids = src_token_ids + [self.tokenizer.pad_token_id] * pad_length
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask
	
	def sample(self, idx: int):
		"""
		Overriding the sample method to return the input for CodeBERT as well.
		:param idx:
		:return:
		"""
		
		_id = self.ids[idx]
		
		# Get the input for CodeBERT
		bert_input_ids, bert_mask = self.get_bert_ip(idx)
		
		# Get the label
		if self.mode == 'train' and len(self.labels) > 0:
			assert _id in self.labels, f"Label not found for {_id}"
			possible_labels: List[int] = self.labels[_id]
			
			# Multi-label: one-hot encode
			label = [1 if i in possible_labels else 0 for i in range(self.num_classes)]
			
			# For weight, same weight for possible labels and 0 for others
			weight = [0.0 for i in range(self.num_classes)]
			if len(possible_labels) > 0:
				weight = [1.0 / len(possible_labels) if i in possible_labels else 0.0 for i in range(self.num_classes)]
			
			label = torch.FloatTensor(label)
			weight = torch.FloatTensor(weight)
			
			return bert_input_ids, bert_mask, label, weight
		
		else:
			return bert_input_ids, bert_mask
	
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
	
	def get_ids(self):
		return self.ids


class MBPP_Dataset_w_PromptEmbedding(LibraryBaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None,
			split: float = 0.5
	):
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.split = split
		
		embedding_path = f'./logging/{mode}_prompt_embeddings.pkl'
		self.prompt_embeddings = torch.load(embedding_path)
		
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_length, mode)
	
	def read_data(self):
		
		# Read the MBPP data
		with open(self.path_to_data, 'r') as f:
			data = f.readlines()
		
		data = [json.loads(d) for d in data]
		
		# Divide into train-test
		if self.mode == 'train':
			data = data[:int(self.split * len(data))]
		elif self.mode == 'test':
			data = data[int(self.split * len(data)):]
		
		partitioned_data = {}
		for problem in tqdm(data, ncols=0, total=len(data),
							desc="Reading MBPP examples from {} (mode = {}): ".format(self.path_to_data,
																					  self.mode)):
			f_id = problem['task_id']
			q_str = problem['prompt']
			a_str = problem['canonical_solution']
			partitioned_data[f_id] = (q_str, a_str)
		
		return partitioned_data
	
	def sample(self, idx: int):
		"""
		Overriding the sample method to return the prompt embedding as well.
		:param idx:
		:return:
		"""
		_id = self.ids[idx]
		
		try:
			prompt_embedding = self.prompt_embeddings[_id]
		except KeyError:
			raise KeyError(f"Prompt embedding not found for {_id}")
		
		prompt_embedding = torch.squeeze(prompt_embedding)
		
		if self.mode == 'train':
			src_input_ids, src_mask, trg_label_ids, trg_mask = super().sample(idx)
			
			return prompt_embedding, src_input_ids, src_mask, trg_label_ids, trg_mask
		
		else:
			src_input_ids, src_mask = super().sample(idx)
			return prompt_embedding, src_input_ids, src_mask


class Sample_Dataset(LibraryBaseDataset):
	
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int,
			max_length: int,
			sample_problems: Union[int, None] = None
	):
		
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_length)
	
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


class APPS_Dataset(LibraryBaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			sample_problems: Union[int, None] = None
	):
		
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		path_to_data = os.path.join(path_to_data, mode)
		
		super().__init__(path_to_data, tokenizer, max_prompt_length, max_length, mode)
	
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
