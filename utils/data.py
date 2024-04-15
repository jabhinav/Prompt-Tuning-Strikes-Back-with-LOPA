import json
import logging
from typing import List, Any, Union

import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
	def __init__(self, path_to_data: str, tokenizer, max_prompt_length, max_length, mode: str,
				 return_dict: bool = False):
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
				new_input_ids = [self.tokenizer.pad_token_id] * self.max_prompt_length
				new_input_ids[-len(src_input_ids):] = src_input_ids
				src_input_ids = new_input_ids
			
			src_input_ids = torch.LongTensor(src_input_ids)
			mask = src_input_ids.ne(self.tokenizer.pad_token_id)
			return src_input_ids, mask
		
		else:
			answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
			answer_token_ids.append(self.tokenizer.eos_token_id)
			
			src_input_ids.extend(answer_token_ids)
			
			# # Want to generate prompt as part of the response [Uncomment]
			# trg_label_ids.extend(question_token_ids)
			
			# # Want to generate response only
			trg_label_ids.extend([-100] * len(question_token_ids))
			
			trg_label_ids.extend(answer_token_ids)
			
			# Cut off the excess
			if len(src_input_ids) > self.max_length:
				# Truncate i/p and label from right (this will auto. truncate only the response)
				src_input_ids = src_input_ids[:self.max_length]
				trg_label_ids = trg_label_ids[:self.max_length]
			
			# # Print the shapes
			# print(f"[Debug] src_input_ids: {len(src_input_ids)}")
			# print(f"[Debug] trg_label_ids: {len(trg_label_ids)}")
			
			return self.process(src_input_ids, trg_label_ids)
	
	def process(self, src_input_ids, trg_label_ids):
		
		# Let's prepare src mask
		src_mask = [1] * len(src_input_ids)  # This will prevent masking out the eos token at the end
		
		if len(src_input_ids) < self.max_length:
			# Pad input [prompt+response] with pad token from left
			new_input_ids = [self.tokenizer.pad_token_id] * self.max_length
			new_input_ids[-len(src_input_ids):] = src_input_ids
			src_input_ids = new_input_ids
			
			# Pad label with -100
			new_label_ids = [-100] * self.max_length
			new_label_ids[-len(trg_label_ids):] = trg_label_ids
			trg_label_ids = new_label_ids
		
		# Pad the src_mask with 0s from left
		src_mask = [0] * (self.max_length - len(src_mask)) + src_mask
		# src_mask = [bool(i) for i in src_mask]
		
		# Convert to tensors
		src_input_ids = torch.LongTensor(src_input_ids)
		trg_label_ids = torch.LongTensor(trg_label_ids)
		src_mask = torch.BoolTensor(src_mask)
		
		# src_mask = src_input_ids.ne(self.tokenizer.pad_token_id)  # mask out padding
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


class MBPP_Dataset(BaseDataset):
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


class MBPP_Dataset_wEnc(BaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			enc_tokenizer=None,
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
		
		self.enc_tokenizer = enc_tokenizer
		
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
	
	def get_enc_ip(self, idx):
		
		if self.enc_tokenizer is None:
			raise ValueError("Encoder tokenizer not provided. Even if not required, provide a dummy tokenizer.")
		
		q_str, _ = self.data[self.ids[idx]]
		src_tokens = self.enc_tokenizer.tokenize(q_str)
		
		# For encoder-only models that are designed for language understanding tasks
		if ((hasattr(self.enc_tokenizer, 'cls_token') and self.enc_tokenizer.cls_token is not None) and
				(hasattr(self.enc_tokenizer, 'sep_token') and self.enc_tokenizer.sep_token is not None)):
			max_bert_length = self.max_prompt_length - 2  # [CLS] and [SEP]
			src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
			src_tokens = [self.enc_tokenizer.cls_token] + src_tokens + [self.enc_tokenizer.sep_token]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from right (should be fine even when encoder is a causal LM)
			pad_length = self.max_prompt_length - len(src_token_ids)
			src_token_ids = src_token_ids + [self.enc_tokenizer.pad_token_id] * pad_length
		
		# For all other models
		else:
			src_tokens = src_tokens[-self.max_prompt_length:]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from left
			pad_length = self.max_prompt_length - len(src_token_ids)
			src_token_ids = [self.enc_tokenizer.pad_token_id] * pad_length + src_token_ids
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.enc_tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask
	
	def sample(self, idx: int):
		"""
		Overriding the sample method to return the input for CodeBERT as well.
		:param idx:
		:return:
		"""
		
		_id = self.ids[idx]
		
		# Get the input for CodeBERT
		enc_input_ids, enc_mask = self.get_enc_ip(idx)
		
		if self.mode == 'train':
			src_input_ids, src_mask, trg_label_ids, trg_mask = super().sample(idx)
			return enc_input_ids, enc_mask, src_input_ids, src_mask, trg_label_ids, trg_mask
		
		else:
			src_input_ids, src_mask = super().sample(idx)
			return enc_input_ids, enc_mask, src_input_ids, src_mask


class MBPP_Dataset_wEnc_Augmented(BaseDataset, IterableDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			enc_tokenizer=None,
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
		
		self.enc_tokenizer = enc_tokenizer
		
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
	
	def get_enc_ip(self, idx):
		
		if self.enc_tokenizer is None:
			raise ValueError("Encoder tokenizer not provided. Even if not required, provide a dummy tokenizer.")
		
		q_str, _ = self.data[self.ids[idx]]
		src_tokens = self.enc_tokenizer.tokenize(q_str)
		
		# For encoder-only models that are designed for language understanding tasks
		if ((hasattr(self.enc_tokenizer, 'cls_token') and self.enc_tokenizer.cls_token is not None) and
				(hasattr(self.enc_tokenizer, 'sep_token') and self.enc_tokenizer.sep_token is not None)):
			max_bert_length = self.max_prompt_length - 2  # [CLS] and [SEP]
			src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
			src_tokens = [self.enc_tokenizer.cls_token] + src_tokens + [self.enc_tokenizer.sep_token]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from right (should be fine even when encoder is a causal LM)
			pad_length = self.max_prompt_length - len(src_token_ids)
			src_token_ids = src_token_ids + [self.enc_tokenizer.pad_token_id] * pad_length
		
		# For all other models
		else:
			src_tokens = src_tokens[-self.max_prompt_length:]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from left
			pad_length = self.max_prompt_length - len(src_token_ids)
			src_token_ids = [self.enc_tokenizer.pad_token_id] * pad_length + src_token_ids
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.enc_tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask

	def __iter__(self):
		
		for idx in range(len(self)):
			_id: str = self.ids[idx]
			
			# Get the input for CodeBERT
			enc_input_ids, enc_mask = self.get_enc_ip(idx)
			src_input_ids, src_mask = super().sample(idx)
			
			yield {
				"enc_input_ids": enc_input_ids,
				"enc_attention_mask": enc_mask,
				"input_ids": src_input_ids,
				"attention_mask": src_mask,
				"task_id": idx  # We can only provide int as task_id. Map later!
			}
			
			
			