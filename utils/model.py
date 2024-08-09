import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from utils.custom import is_rank_0
from utils.xformer import load_base_model, get_huggingface_path


def compute_grad_norm(model):
	total_norm: float = 0.
	for name, p in model.named_parameters():
		if p.grad is not None and p.requires_grad:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
	total_norm = total_norm ** (1. / 2)
	return total_norm


def get_response_embedding(model, prompt, prompt_mask, response):
	# Do forward pass and get the response embedding corresponding to the last layer and last token
	resp_embeddings = model(
		input_ids=prompt,
		attention_mask=prompt_mask,
		labels=response,
		output_hidden_states=True
	)['hidden_states'][-1]
	
	# Last token embedding
	resp_embedding = resp_embeddings[:, -1, :]
	
	return resp_embedding


def get_clf_embedding(args, model, prompt, prompt_mask, response, library_idx):
	clf_embedding = model(
		library_idx=library_idx,
		input_ids=prompt,
		attention_mask=prompt_mask,
		labels=response,
		output_hidden_states=True
	)['hidden_states'][-1]
	
	# Last virtual token embedding (this won't change no matter the prompt)
	embedding = clf_embedding[:, args.num_virtual_tokens - 1, :]
	
	return embedding


class LatentPromptAttentionGenerator(torch.nn.Module):
	"""
	Used in LOPA to generate the instance-specific attention weights, Z_I before the gating function.
	"""
	
	def __init__(self, args, n_virtual_tokens, word_embedding_dim, use_bias=True, freeze_base=True, MLP_h=None):
		super(LatentPromptAttentionGenerator, self).__init__()
		
		config, base = load_base_model(
			model_type=args.enc_model_type,
			config_name=get_huggingface_path(args.enc_model_type),
			model_path=get_huggingface_path(args.enc_model_type),
		)
		self.args = args
		self.config = config
		self.base = base
		self.freeze_base = freeze_base
		self.rank = self.args.lp_rank
		
		# # Base model does not require any training - freeze the weights
		if self.freeze_base:
			if is_rank_0():
				print("[INFO] Freezing the Enc base model weights")
			for param in self.base.parameters():
				param.requires_grad = False
		else:
			if is_rank_0():
				print("[INFO] Tuning the Enc base model weights")
		
		# For each virtual token, predict a word embedding - mu and logvar
		self.n_virtual_tokens = n_virtual_tokens
		self.word_embedding_dim = word_embedding_dim
		
		# Set params
		dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config, 'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding the row vectors - weighs virtual tokens
		self.row_dropout = torch.nn.Dropout(dropout_prob)
		self.row_dense = torch.nn.Linear(hidden_dim, MLP_h, bias=use_bias)
		self.row_out_proj = torch.nn.Linear(MLP_h, n_virtual_tokens * self.rank, bias=use_bias)
		
		# Define the head for encoding the column vectors - weighs the word embedding dimensions
		self.col_dropout = torch.nn.Dropout(dropout_prob)
		self.col_dense = torch.nn.Linear(hidden_dim, MLP_h, bias=use_bias)
		self.col_out_proj = torch.nn.Linear(MLP_h, word_embedding_dim * self.rank, bias=use_bias)
		
		self.init_predictor_head()
	
	def init_predictor_head(self):
		# Initialize the weights for the row head
		self.row_dense.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.row_out_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.row_out_proj.bias.data.zero_()
		
		# Initialize the weights for the column head
		self.col_dense.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.col_out_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.col_out_proj.bias.data.zero_()
	
	def get_instance_embedding(self, input_ids, attention_mask=None):
		
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type in ['codebert-base', 'roberta-base', 'roberta-large']:
			# [IMP] Codebert Base is based on RoBERTa model: Getting the seq representation should match with that of
			# default RobertaForSequenceClassification & RobertaClassificationHead
			x = self.base(input_ids, attention_mask=attention_mask)
			x = x[0][:, 0, :]
		
		elif self.args.enc_model_type == 'codet5p-110m-embedding':
			x = self.base(input_ids, attention_mask=attention_mask)
		
		elif self.args.enc_model_type in ['codesage-base', 'codesage-small', 'codesage-large']:
			x = self.base(input_ids, attention_mask=attention_mask, return_dict=True)
			x = torch.nn.functional.normalize(x.pooler_output, p=2, dim=1)
		
		else:
			# Should be used for decoder models
			x = self.base(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=input_ids,  # TODO: See if the forward pass works without labels
				output_hidden_states=True
			)['hidden_states'][-1]  # Get the last layer hidden states
			
			# Last token embedding or [TODO] pass seq_lengths to get last non-padded token embedding
			x = x[:, -1, :]
		# x = x[torch.arange(x.size(0)), seq_lengths, :]
		
		return x
	
	def forward(self, input_ids, attention_mask=None):
		
		if self.freeze_base:
			with torch.no_grad():
				# Get the instance embedding
				x = self.get_instance_embedding(input_ids, attention_mask)
				x = x.detach()
		
		else:
			# Get the instance embedding
			x = self.get_instance_embedding(input_ids, attention_mask)
		
		# Predict the row weights
		row_weights = self.row_dropout(x)
		row_weights = self.row_dense(row_weights)
		row_weights = torch.nn.functional.tanh(row_weights)
		row_weights = self.row_out_proj(row_weights)
		row_weights = row_weights.view(-1, self.n_virtual_tokens, self.rank)
		
		# Predict the column weights
		col_weights = self.col_dropout(x)
		col_weights = self.col_dense(col_weights)
		col_weights = torch.nn.functional.tanh(col_weights)
		col_weights = self.col_out_proj(col_weights)
		col_weights = col_weights.view(-1, self.word_embedding_dim, self.rank)
		
		# [Older, for r=1] Multiply: uk ∈ R^l, vk ∈ R^d -> uk * vk^T ∈ R^(l x d)
		# prompt_specific_clf_embedding = torch.einsum('bi,bj->bij', row_weights, col_weights)
		
		# [Latest, for r>=1] Multiply: uk ∈ R^l x r, vk ∈ R^d x r -> uk * vk^T ∈ R^(l x d)
		prompt_specific_clf_embedding = torch.einsum('bir,bjr->bij', row_weights, col_weights)
		
		return prompt_specific_clf_embedding
	
	def __str__(self):
		return f"MyEncoder/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"MyEncoder/{self.args.enc_model_type}"


class IDPGSoftPromptGenerator(torch.nn.Module):
	def __init__(self, args, n_virtual_tokens, word_embedding_dim, use_bias=True, MLP_h=None):
		super(IDPGSoftPromptGenerator, self).__init__()
		
		# In IDPG, the encoder is the same as the decoder
		config, base = load_base_model(
			model_type=args.enc_model_type,
			config_name=get_huggingface_path(args.enc_model_type),
			model_path=get_huggingface_path(args.enc_model_type),
		)
		self.args = args
		self.config = config
		self.base = base  # Could be any model
		
		# Base model does not require any training - freeze the weights
		for param in self.base.parameters():
			param.requires_grad = False
		
		# For each virtual token, predict a word embedding - mu and logvar
		self.n_virtual_tokens = n_virtual_tokens
		self.word_embedding_dim = word_embedding_dim
		
		# Set params [Should be same as the model used for the base]
		dropout_prob = config.hidden_dropout_prob if hasattr(config,  'hidden_dropout_prob') else config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config, 'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding all virtual tokens
		self._dropout = torch.nn.Dropout(dropout_prob)
		self.layer_down_project = torch.nn.Linear(hidden_dim, MLP_h, bias=use_bias)
		self.layer_up_project = torch.nn.Linear(MLP_h, n_virtual_tokens * word_embedding_dim, bias=use_bias)
		
		self.init_predictor_head()
	
	def init_predictor_head(self):
		# Initialize the weights for the row head
		self.layer_down_project.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.layer_up_project.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.layer_up_project.bias.data.zero_()
	
	@torch.no_grad()
	def get_instance_embedding(self, input_ids, attention_mask=None):
		
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type in ['codebert-base', 'roberta-base', 'roberta-large']:
			x = self.base(input_ids, attention_mask=attention_mask)
			x = x[0][:, 0, :]
		
		elif self.args.enc_model_type == 'codet5p-110m-embedding':
			x = self.base(input_ids, attention_mask=attention_mask)
		
		elif self.args.enc_model_type in ['codesage-base', 'codesage-small', 'codesage-large']:
			x = self.base(input_ids, attention_mask=attention_mask, return_dict=True)
			x = torch.nn.functional.normalize(x.pooler_output, p=2, dim=1)
		
		else:
			# Should be used for decoder models
			x = self.base(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=input_ids,  # TODO: See if the forward pass works without labels
				output_hidden_states=True
			)['hidden_states'][-1]
			
			# Last token embedding or [TODO] pass seq_lengths to get last non-padded token embedding
			x = x[:, -1, :]
		# x = x[torch.arange(x.size(0)), seq_lengths, :]
		
		return x.detach()
	
	def forward(self, input_ids, attention_mask=None):
		
		inst_embedding = self.get_instance_embedding(input_ids, attention_mask)
		
		# Predict the row weights
		soft_prompt_embedding = self._dropout(inst_embedding)
		soft_prompt_embedding = self.layer_down_project(soft_prompt_embedding)
		soft_prompt_embedding = torch.nn.functional.tanh(soft_prompt_embedding)
		soft_prompt_embedding = self.layer_up_project(soft_prompt_embedding)
		
		# Reshape [B, N * D] -> [B, N, D]
		soft_prompt_embedding = soft_prompt_embedding.view(-1, self.n_virtual_tokens, self.word_embedding_dim)
		return soft_prompt_embedding
	
	def __str__(self):
		return f"IDPGEncoder/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"IDPGEncoder/{self.args.enc_model_type}"


class PHMLayer(nn.Module):
	
	def __init__(self, n, in_features, out_features):
		super(PHMLayer, self).__init__()
		
		assert out_features % n == 0, "out_features should be divisible by n"
		assert in_features % n == 0, "in_features should be divisible by n"
		
		self.n = n
		self.in_features = in_features
		self.out_features = out_features
		
		self.bias = Parameter(torch.Tensor(out_features))
		
		self.a = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
		
		self.s = Parameter(
			torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features // n, self.in_features // n))))
		
		self.weight = torch.zeros((self.out_features, self.in_features))
		
		fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.bias, -bound, bound)
	
	def kronecker_product1(self, a, b):  # adapted from Bayer Research's implementation
		siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
		res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
		siz0 = res.shape[:-4]
		out = res.reshape(siz0 + siz1)
		return out
	
	def forward(self, input: Tensor) -> Tensor:
		self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
		input = input.type(dtype=self.weight.type())
		return F.linear(input, weight=self.weight, bias=self.bias)
	
	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None)
	
	def reset_parameters(self) -> None:
		init.kaiming_uniform_(self.a, a=math.sqrt(5))
		init.kaiming_uniform_(self.s, a=math.sqrt(5))
		fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.bias, -bound, bound)


class IDPGSoftPromptGenerator_wPHM(torch.nn.Module):
	def __init__(self, args, use_bias=True, MLP_h=None, n=16):
		super(IDPGSoftPromptGenerator_wPHM, self).__init__()
		
		config, base = load_base_model(
			args,
			model_type=args.enc_model_type,
			model_name_or_path=get_huggingface_path(args.enc_model_type)
		)
		
		self.args = args
		self.config = config
		self.base = base
		
		# Base model does not require any training - freeze the weights
		for param in self.base.parameters():
			param.requires_grad = False
		
		# For each virtual token, predict the embedding
		self.config.n_virtual_tokens = self.args.total_virtual_tokens
		self.config.word_embedding_dim = self.args.word_embedding_dim
		
		# Set params [Should be same as the model used for the base]
		dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config, 'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding all virtual tokens
		self._dropout = torch.nn.Dropout(dropout_prob)
		self.layer_down_project = PHMLayer(in_features=hidden_dim, out_features=MLP_h, n=n)
		self.layer_up_project = PHMLayer(in_features=MLP_h,
										 out_features=self.config.n_virtual_tokens * self.config.word_embedding_dim,
										 n=n)
	
	@torch.no_grad()
	def get_instance_embedding(self, input_ids, attention_mask=None, token_type_ids=None):
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type == 'roberta-large':
			x = self.base(
				input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
			x = x[0]
			x = x[:, 0, :]  # take <s> which is the first token as seq. representation (equiv. to [CLS])
		else:
			raise NotImplementedError
		return x.detach()
	
	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		
		inst_embedding = self.get_instance_embedding(input_ids, attention_mask, token_type_ids)
		
		# Predict the row weights
		soft_prompt_embedding = self._dropout(inst_embedding)
		soft_prompt_embedding = self.layer_down_project(soft_prompt_embedding)
		soft_prompt_embedding = torch.nn.functional.tanh(soft_prompt_embedding)
		soft_prompt_embedding = self.layer_up_project(soft_prompt_embedding)
		
		# Reshape [B, N * D] -> [B, N, D]
		soft_prompt_embedding = soft_prompt_embedding.view(-1, self.config.n_virtual_tokens,
														   self.config.word_embedding_dim)
		return soft_prompt_embedding
	
	def __str__(self):
		return f"IDPG/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"IDPG/{self.args.enc_model_type}"


class LOPA(torch.nn.Module):
	
	def __init__(self, enc, dec):
		super(LOPA, self).__init__()
		self.enc = enc
		self.dec = dec
		
		# Beta
		self.config = self.enc.base.config
	
	def forward(self, batch):
		# Encode
		att_logits = self.enc(
			input_ids=batch['enc_input_ids'],
			attention_mask=batch['enc_attention_mask'],
		)
		att_weights = torch.sigmoid(att_logits)
		
		# Decode
		output = self.dec(
			latent_attention_weights=att_weights,
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			labels=batch['labels'],
			output_hidden_states=True
		)
		
		return output


class IDPG(torch.nn.Module):
	
	def __init__(self, enc, dec):
		super(IDPG, self).__init__()
		self.enc = enc
		self.dec = dec
		
		# Beta
		self.config = self.enc.base.config
	
	def forward(self, batch):
		# Encode
		soft_prompt = self.enc(
			input_ids=batch['enc_input_ids'],
			attention_mask=batch['enc_attention_mask'],
		)
		
		# Decode
		output = self.dec(
			soft_prompt=soft_prompt,
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			labels=batch['labels'],
			output_hidden_states=True
		)
		
		return output
