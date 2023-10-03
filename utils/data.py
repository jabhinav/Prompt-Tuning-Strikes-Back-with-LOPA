import os
from tqdm import tqdm
import torch
from typing import List, Tuple, Dict, Union, Any, Optional
from torch.utils.data import Dataset


class LibraryBaseDataset(Dataset):
	def __init__(self, path_to_data: str):
		self.path_to_data = path_to_data
		self.data = self.read_data()
		
	def read_data(self):
		raise NotImplementedError
	
	def sample(self, idx: int):
		raise NotImplementedError
	
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
	
	def __init__(self, path_to_data: str, tokenizer: Any, max_length: int = 512):
		super().__init__(path_to_data)
		
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.ids: List[str] = list(self.data.keys())
	
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
		
		# Pad src with eos on right (pad tokens won't be attended to by the model)
		if len(src_input_ids) < self.max_length:
			new_src_input_ids = [self.tokenizer.eos_token_id] * self.max_length
			new_src_input_ids[:len(src_input_ids)] = src_input_ids
			src_input_ids = new_src_input_ids
			
			# Pad trg label with -100 on right (CrossEntropyLoss ignores -100 by default)
			new_trg_label_ids = [-100] * self.max_length
			new_trg_label_ids[:len(trg_label_ids)] = trg_label_ids
			trg_label_ids = new_trg_label_ids
		
		src_input_ids = src_input_ids[:self.max_length]
		trg_label_ids = trg_label_ids[:self.max_length]
		
		# Convert to tensors
		src_input_ids = torch.LongTensor(src_input_ids)
		trg_label_ids = torch.LongTensor(trg_label_ids)
		
		src_mask = src_input_ids.ne(self.tokenizer.eos_token_id)  # mask out padding
		trg_mask = trg_label_ids.ne(-100)  # mask out padding
		
		return src_input_ids, src_mask, trg_label_ids, trg_mask
