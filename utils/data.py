import json
import logging
import os
from typing import List, Any, Dict, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import utils.cruxeval_tasks as cruxeval_tasks
from utils.cruxeval_tasks.base import Task

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
	def __init__(self, path_to_data: str, tokenizer, max_prompt_length, max_length, mode: str,
				 return_dict: bool = False):
		self.path_to_data = path_to_data
		
		assert os.path.exists(self.path_to_data), \
			f"Path to data does not exist: {self.path_to_data}. Load here or provide a valid path."
		
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


class MBPP_Dataset_wEnc(BaseDataset):
	def __init__(
			self,
			path_to_data: str,
			tokenizer: Any,
			max_prompt_length: int = 512,
			max_length: int = 512,
			mode: str = 'train',
			enc_tokenizer=None,
			split: float = 0.5,
			finer_split: float = 1.0,
			use_first_half: bool = True,
	):
		# # Initialize mode before calling super().__init__
		# max_length=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024 <- From APPS github repo
		# self.sample_problems = sample_problems  # Number of problems to sample from the dataset
		self.split = split
		
		# Use the first <finer_split> fraction of the data as training-1 and the rest as training-2
		self.finer_split = finer_split
		self.use_first_half = use_first_half
		
		self.max_enc_length = 512  # set this based on the encoder tokenizer
		
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
			max_bert_length = self.max_enc_length - 2  # [CLS] and [SEP]
			src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
			src_tokens = [self.enc_tokenizer.cls_token] + src_tokens + [self.enc_tokenizer.sep_token]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from right (should be fine even when encoder is a causal LM)
			pad_length = self.max_enc_length - len(src_token_ids)
			src_token_ids = src_token_ids + [self.enc_tokenizer.pad_token_id] * pad_length
		
		# For all other models
		else:
			src_tokens = src_tokens[-self.max_enc_length:]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from left
			pad_length = self.max_enc_length - len(src_token_ids)
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
			return {
				"enc_input_ids": enc_input_ids,
				"enc_attention_mask": enc_mask,
				"input_ids": src_input_ids,
				"attention_mask": src_mask,
				"labels": trg_label_ids,
				"task_id": idx  # We can only provide int as task_id. Map later!
			}
		
		else:
			src_input_ids, src_mask = super().sample(idx)
			return {
				"enc_input_ids": enc_input_ids,
				"enc_attention_mask": enc_mask,
				"input_ids": src_input_ids,
				"attention_mask": src_mask,
				"task_id": idx  # We can only provide int as task_id. Map later!
			}
	
	def __iter__(self):
		
		for idx in range(len(self)):
			_id: str = self.ids[idx]
			
			# Get the input for CodeBERT
			enc_input_ids, enc_mask = self.get_enc_ip(idx)
			
			if self.mode == 'train':
				src_input_ids, src_mask, trg_label_ids, trg_mask = super().sample(idx)
				yield {
					"enc_input_ids": enc_input_ids,
					"enc_attention_mask": enc_mask,
					"input_ids": src_input_ids,
					"attention_mask": src_mask,
					"labels": trg_label_ids,
					"task_id": idx  # We can only provide int as task_id. Map later!
				}
			
			else:
				src_input_ids, src_mask = super().sample(idx)
				yield {
					"enc_input_ids": enc_input_ids,
					"enc_attention_mask": enc_mask,
					"input_ids": src_input_ids,
					"attention_mask": src_mask,
					"task_id": idx  # We can only provide int as task_id. Map later!
				}


class CruxEval_Dataset_wEnc(Dataset):
	"""
	Tokenize and preprocess the dataset
	    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
	    The prompt can either be:
	    - one prompt: normal code completion
	    - two prompts: for infilling mode (prefix, suffix) or instructing-tuning mode (instruction, context)
	"""
	
	def __init__(self, cruxeval_task, tokenizer, enc_tokenizer, max_length, prefix, cot, mode='train'):
		
		self.tokenizer = tokenizer
		self.enc_tokenizer = enc_tokenizer
		self.max_length = max_length
		self.max_enc_length = 512  # set this based on the encoder tokenizer
		
		self.prefix = prefix
		self.cot = cot
		if self.cot:
			raise ValueError("COT based training or testing with this dataloader is not supported for CruxEval tasks.")
		self.mode = mode
		
		# Get the task
		self.task_name: str = cruxeval_task
		self.task: Task = cruxeval_tasks.get_task(self.task_name, cot=self.cot)
		
		# Get the data
		self.data = self.get_data()
		logger.info(f"[INFO] Loaded the CruxEval task: {self.task_name}")
		print(f"[INFO] Loaded the CruxEval task: {self.task_name}")
		
		# Get the mapping from the row index to the task_id. Ex: {400: 'sample_400'}
		self.idx_to_id = self.get_index_to_id_mapping()
		
		# Get the mapping from sample position in the dataset to the row index
		self.idx_to_pos = self.get_index_to_position_mapping()
		
		# Show stats
		self.show_stats()
		
		# Convert examples to features
		self.inputs, self.labels, self.attn_masks, self.enc_inputs, self.enc_masks, self.row_idxs, self.input_lens = self.convert_examples_to_features()
	
	def get_data(self):
		
		dataset = self.task.get_dataset()
		dataset_rows = range(dataset.num_rows)
		dataset = dataset.add_column("row_index", dataset_rows)
		
		if self.mode == 'train':
			# Get first half of the data
			dataset = dataset.filter(lambda example: example['row_index'] < len(dataset) // 2)
		else:
			# Get second half of the data
			dataset = dataset.filter(lambda example: example['row_index'] >= len(dataset) // 2)
		
		return dataset
	
	def show_stats(self):
		logger.info(f"[INFO] Number of rows in the dataset: {len(self.data)}")
		
		prompts = []
		references = []
		for i in range(len(self)):
			sample = self.data[i]
			prompt_contents = self.task.get_prompt(sample)
			assert isinstance(prompt_contents, str)
			prompt = self.prefix + prompt_contents
			prompts.append(prompt)
			
			if not self.cot:
				reference = self.task.get_label(sample)
				references.append(reference)
		
		inputs = self.tokenizer(
			prompts,
			padding=True,
			truncation=True,
			return_tensors="pt",
			max_length=self.max_length,
		)
		input_lens = [inputs.attention_mask[i].sum() for i in range(len(self))]
		logger.info(
			f"[INFO] Prompt Length Stats: Min: {min(input_lens)}, Max: {max(input_lens)}, Avg: {sum(input_lens) / len(input_lens)}")
		
		if len(references) > 0:
			outputs = self.tokenizer(
				references,
				padding=True,
				truncation=True,
				return_tensors="pt",
				max_length=self.max_length,
			)
			output_lens = [outputs.attention_mask[i].sum() for i in range(len(self))]
			logger.info(
				f"[INFO] Reference Length Stats: Min: {min(output_lens)}, Max: {max(output_lens)}, Avg: {sum(output_lens) / len(output_lens)}")
	
	def get_index_to_id_mapping(self):
		# Get the mapping from the index to the task_id
		idx_to_id = {}
		for i in range(len(self)):
			sample = self.data[i]
			idx_to_id[sample["row_index"]] = sample["id"]
		
		return idx_to_id
	
	def get_index_to_position_mapping(self):
		# Get the mapping from the position in the dataset to the row index
		idx_to_pos = {}
		for i in range(len(self)):
			sample = self.data[i]
			idx_to_pos[sample["row_index"]] = i
		
		return idx_to_pos
	
	def get_enc_ip(self, q_str):
		
		if self.enc_tokenizer is None:
			raise ValueError("Encoder tokenizer not provided. Even if not required, provide a dummy tokenizer.")
		
		src_tokens = self.enc_tokenizer.tokenize(q_str)
		
		# For encoder-only models that are designed for language understanding tasks
		if ((hasattr(self.enc_tokenizer, 'cls_token') and self.enc_tokenizer.cls_token is not None) and
				(hasattr(self.enc_tokenizer, 'sep_token') and self.enc_tokenizer.sep_token is not None)):
			max_bert_length = self.max_enc_length - 2  # [CLS] and [SEP]
			src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
			src_tokens = [self.enc_tokenizer.cls_token] + src_tokens + [self.enc_tokenizer.sep_token]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from right (should be fine even when encoder is a causal LM)
			pad_length = self.max_enc_length - len(src_token_ids)
			src_token_ids = src_token_ids + [self.enc_tokenizer.pad_token_id] * pad_length
		
		# For all other models
		else:
			src_tokens = src_tokens[-self.max_enc_length:]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from left
			pad_length = self.max_enc_length - len(src_token_ids)
			src_token_ids = [self.enc_tokenizer.pad_token_id] * pad_length + src_token_ids
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.enc_tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask
	
	def convert_examples_to_features(self):
		inputs, labels, attn_masks, enc_inputs, enc_masks, row_idxs, input_lens = [], [], [], [], [], [], []
		
		pbar = tqdm(total=len(self), ncols=0, desc=f"Converting examples to features (mode={self.mode}): ")
		for i in range(len(self)):
			sample = self.data[i]
			label_contents = self.task.get_label(sample)
			assert isinstance(label_contents, str)
			
			row_idxs.append(sample["row_index"])
			
			label = self.prefix + label_contents
			splits = label.split("[ANSWER]")
			
			prompt = "[ANSWER]".join(splits[:-1]) + "[ANSWER]"
			target = splits[-1]
			
			# ########################### Prepare for the base model ########################### #
			
			# Tokenize
			prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, verbose=False, add_special_tokens=False)
			target_ids = self.tokenizer.encode(target, verbose=False, add_special_tokens=False) + [self.tokenizer.eos_token_id]
			
			# Prepare input
			if self.mode == 'train':
				input_ids = prompt_ids + target_ids
				# Prepare label
				label_ids = [-100] * len(prompt_ids) + target_ids[:-1] + [-100]  # -100 is the ignore index
			else:
				input_ids = prompt_ids
				label_ids = None
			
			# Prepare attention mask
			attn_mask = [1] * len(input_ids)
			
			# Pad from right
			if len(input_ids) < self.max_length:
				new_input_ids = [self.tokenizer.pad_token_id] * self.max_length
				new_input_ids[:len(input_ids)] = input_ids
				input_ids = new_input_ids
				
				attn_mask += [0] * (self.max_length - len(attn_mask))
				
				if label_ids is not None:
					new_label_ids = [-100] * self.max_length
					new_label_ids[:len(label_ids)] = label_ids
					label_ids = new_label_ids
			
			# Truncate from left
			input_ids = input_ids[-self.max_length:]
			label_ids = label_ids[-self.max_length:] if label_ids is not None else None
			attn_mask = attn_mask[-self.max_length:]
			
			# Convert to tensors
			input_ids = torch.LongTensor(input_ids)
			label_ids = torch.LongTensor(label_ids) if label_ids is not None else None
			attn_mask = torch.BoolTensor(attn_mask)
			
			# ########################### Prepare for the encoder model ########################### #
			enc_input_ids, enc_mask = self.get_enc_ip(prompt)
			
			# Append to the lists
			inputs.append(input_ids)
			labels.append(label_ids)
			attn_masks.append(attn_mask)
			enc_inputs.append(enc_input_ids)
			enc_masks.append(enc_mask)
			input_lens.append(sum(attn_mask))
			
			pbar.update()
		
		return inputs, labels, attn_masks, enc_inputs, enc_masks, row_idxs, input_lens
	
	def __len__(self):
		return self.data.num_rows
	
	def __getitem__(self, i):
		if self.mode == 'train':
			return {
				"task_id": self.row_idxs[i],
				"input_ids": self.inputs[i],
				"attention_mask": self.attn_masks[i],
				"labels": self.labels[i],
				"enc_input_ids": self.enc_inputs[i],
				"enc_attention_mask": self.enc_masks[i],
				"input_len": self.input_lens[i]
			}
		else:
			return {
				"task_id": self.row_idxs[i],
				"input_ids": self.inputs[i],
				"attention_mask": self.attn_masks[i],
				"enc_input_ids": self.enc_inputs[i],
				"enc_attention_mask": self.enc_masks[i],
				"input_len": self.input_lens[i]
			}
	
	def __iter__(self):
		for i in range(len(self)):
			if self.mode == 'train':
				yield {
					"task_id": self.row_idxs[i],
					"input_ids": self.inputs[i],
					"attention_mask": self.attn_masks[i],
					"labels": self.labels[i],
					"enc_input_ids": self.enc_inputs[i],
					"enc_attention_mask": self.enc_masks[i],
					"input_len": self.input_lens[i]
				}
			else:
				yield {
					"task_id": self.row_idxs[i],
					"input_ids": self.inputs[i],
					"attention_mask": self.attn_masks[i],
					"enc_input_ids": self.enc_inputs[i],
					"enc_attention_mask": self.enc_masks[i],
					"input_len": self.input_lens[i]
				}


class NLG_Dataset_wEnc(Dataset):
	"""
		For following Datasets:
		"The E2E dataset: New challenges for end-to-end generation." arXiv preprint arXiv:1706.09254 (2017).
		"The WebNLG challenge: Generating text from RDF data." ACL Anthology, 2017.
		"Dart: Open-domain structured data record to text generation." arXiv preprint arXiv:2007.02871 (2020).

	"""
	def __init__(self,
				 path_to_data: str,
				 tokenizer: Any,
				 max_length: int,
				 mode: str = 'train',
				 enc_tokenizer=None,
				 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0,
				 prefix_cursor=1000000, infix_cursor=2000000
				 ):
		self.path_to_data = path_to_data
		self.tokenizer = tokenizer
		self.max_enc_length = 256  # set this based on the encoder tokenizer and data stats
		self.max_length = max_length
		self.mode = mode
		self.enc_tokenizer = enc_tokenizer
		
		self.max_eval_length = max_eval_length
		self.joint_lm = joint_lm
		self.prefix_len = prefix_len
		self.infix_len = infix_len
		self.prefix_cursor = prefix_cursor
		self.infix_cursor = infix_cursor
		
		# Read Data
		self.data, self.aux_info = self.read_data()
		self.orig_ids: List[str] = list(self.data.keys())
		
		# # Get the mapping from the row index to the task_id. Ex: {400: 'sample_400'}
		# self.idx_to_id = self.get_index_to_id_mapping()
		#
		# # Get the mapping from sample position in the dataset to the row index
		# self.idx_to_pos = self.get_index_to_position_mapping()
		
		# Convert examples to features
		self.inputs, self.labels, self.attn_masks, self.enc_inputs, self.enc_masks, self.task_idxs, self.input_lens = self.convert_examples_to_features()
		
	def read_data(self):
		# Read the data from the jsonl file
		with open(self.path_to_data, 'r') as f:
			raw_data = f.readlines()
		
		raw_data = [json.loads(d) for d in raw_data]

		data = {}
		# Some datasets have additional information like seen category indicator in WebNLG and DART
		aux_info = {}
		for idx, problem in tqdm(enumerate(raw_data), ncols=0, total=len(raw_data),
							desc="Reading examples from {} (mode = {}): ".format(self.path_to_data,
																					  self.mode)):
			orig_id = str(idx)
			q_str = problem['context']
			a_str = problem['completion']
			data[orig_id] = (q_str, a_str)
			
			
			if 'cate' in problem:
				aux_info[orig_id] = problem['cate']
		
		return data, aux_info
	
	def get_sample(self, task_idx):
		orig_id = self.orig_ids[task_idx]
		return self.data[orig_id]
	
	def category_seen(self, task_idx) -> bool:
		orig_id = self.orig_ids[task_idx]
		if not orig_id in self.aux_info:
			return None
		return self.aux_info[orig_id]
	
	def get_enc_ip(self, q_str):
		
		if self.enc_tokenizer is None:
			raise ValueError("Encoder tokenizer not provided. Even if not required, provide a dummy tokenizer.")
		
		src_tokens = self.enc_tokenizer.tokenize(q_str)
		
		# For encoder-only models that are designed for language understanding tasks
		if ((hasattr(self.enc_tokenizer, 'cls_token') and self.enc_tokenizer.cls_token is not None) and
				(hasattr(self.enc_tokenizer, 'sep_token') and self.enc_tokenizer.sep_token is not None)):
			max_bert_length = self.max_enc_length - 2  # [CLS] and [SEP]
			src_tokens = src_tokens[-max_bert_length:]  # Truncate the prompt from left
			src_tokens = [self.enc_tokenizer.cls_token] + src_tokens + [self.enc_tokenizer.sep_token]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from right (should be fine even when encoder is a causal LM)
			pad_length = self.max_enc_length - len(src_token_ids)
			src_token_ids = src_token_ids + [self.enc_tokenizer.pad_token_id] * pad_length
		
		# For all other models
		else:
			src_tokens = src_tokens[-self.max_enc_length:]
			src_token_ids = self.enc_tokenizer.convert_tokens_to_ids(src_tokens)
			
			# Pad from left
			pad_length = self.max_enc_length - len(src_token_ids)
			src_token_ids = [self.enc_tokenizer.pad_token_id] * pad_length + src_token_ids
		
		# Convert to tensors
		src_token_ids = torch.LongTensor(src_token_ids)
		src_mask = src_token_ids.ne(self.enc_tokenizer.pad_token_id)  # mask out padding
		
		return src_token_ids, src_mask
	
	def convert_examples_to_features(self):
		inputs, labels, attn_masks, enc_inputs, enc_masks, task_idxs, input_lens = [], [], [], [], [], [], []
		
		pbar = tqdm(total=len(self), ncols=0, desc=f"Converting examples to features (mode={self.mode}): ")
		for idx in range(len(self)):
			
			# ########################### Prepare for the base model ########################### #
			orig_id = self.orig_ids[idx]
			q_str, a_str = self.data[orig_id]
			task_idxs.append(idx)
			
			# Tokenize. context and completion is delimited by " "
			context = self.tokenizer.encode(q_str, verbose=False, add_special_tokens=False) + [self.tokenizer.bos_token_id]
			completion = self.tokenizer.encode(' ' + a_str, verbose=False, add_special_tokens=False) + [self.tokenizer.eos_token_id]
			
			pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)]  # Prefix tokens
			intokens = [i + self.infix_cursor for i in range(0, self.infix_len)]  # Infix tokens

			conditions = pretokens + context + intokens
			
			# Prepare input. _input_len includes length of input with bos and eos tokens(if not test)
			if self.mode == 'train' or self.mode == 'valid':
				_input, _input_len = self.padding_tokens(conditions + completion, self.max_length, self.tokenizer.pad_token_id, 1)
				# Prepare label
				_target = [-100] * len(conditions) + completion
				_target, _ = self.padding_tokens(_target, self.max_length, -100, 1)
			else:
				# During test, max_context_length is max_new_tokens. So we make sure there is some room for generation by truncating from the prompt from left.
				_input, _input_len = self.padding_tokens(conditions, self.max_length, self.tokenizer.pad_token_id, -1, max_context_length=self.max_length - self.max_eval_length)
				_target = None
				
			# Prepare attention mask
			attn_mask = [1] * _input_len
			attn_mask, _ = self.padding_tokens(attn_mask, self.max_length, 0, 1)
			
			# Convert to tensors
			input_ids = torch.LongTensor(_input)
			label_ids = torch.LongTensor(_target) if _target is not None else None
			attn_mask = torch.BoolTensor(attn_mask)
			
			# ########################### Prepare for the encoder model ########################### #
			enc_input_ids, enc_mask = self.get_enc_ip(q_str)
			
			# Append to the lists
			inputs.append(input_ids)
			labels.append(label_ids)
			attn_masks.append(attn_mask)
			enc_inputs.append(enc_input_ids)
			enc_masks.append(enc_mask)
			input_lens.append(_input_len)
			
			pbar.update()
		
		logger.info(f"[INFO] Input Length Stats: Min: {min(input_lens)}, Max: {max(input_lens)}, Avg: {sum(input_lens) / len(input_lens)}")
		
		return inputs, labels, attn_masks, enc_inputs, enc_masks, task_idxs, input_lens

	@staticmethod
	def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):
		
		if max_context_length == 0:
			max_context_length = max_seq_length
		
		if len(tokens) > max_context_length:
			if direct > 0:
				# Truncate from right
				pad_tokens = tokens[:max_context_length]
			else:
				# Truncate from left
				pad_tokens = tokens[-max_context_length:]
		else:
			pad_tokens = tokens
		token_len = len(pad_tokens)
		# Pad from right
		pad_tokens += [pad_token for _ in range(max_seq_length - token_len)]
		return pad_tokens, token_len
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, i):
		if self.mode == 'train' or self.mode == 'valid':
			return {
				"task_id": self.task_idxs[i],
				"input_ids": self.inputs[i],
				"attention_mask": self.attn_masks[i],
				"labels": self.labels[i],
				"enc_input_ids": self.enc_inputs[i],
				"enc_attention_mask": self.enc_masks[i],
				"input_len": self.input_lens[i]
			}
		else:
			return {
				"task_id": self.task_idxs[i],
				"input_ids": self.inputs[i],
				"attention_mask": self.attn_masks[i],
				"enc_input_ids": self.enc_inputs[i],
				"enc_attention_mask": self.enc_masks[i],
				"input_len": self.input_lens[i]
			}
	
	def __iter__(self):
		for i in range(len(self)):
			if self.mode == 'train' or self.mode == 'valid':
				yield {
					"task_id": self.task_idxs[i],
					"input_ids": self.inputs[i],
					"attention_mask": self.attn_masks[i],
					"labels": self.labels[i],
					"enc_input_ids": self.enc_inputs[i],
					"enc_attention_mask": self.enc_masks[i],
					"input_len": self.input_lens[i]
				}
			else:
				yield {
					"task_id": self.task_idxs[i],
					"input_ids": self.inputs[i],
					"attention_mask": self.attn_masks[i],
					"enc_input_ids": self.enc_inputs[i],
					"enc_attention_mask": self.enc_masks[i],
					"input_len": self.input_lens[i]
				}