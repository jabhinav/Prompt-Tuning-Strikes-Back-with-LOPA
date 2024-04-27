import torch

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
	def __init__(self, args, n_virtual_tokens, word_embedding_dim, freeze_base=False):
		super(LatentPromptAttentionGenerator, self).__init__()
		
		config, base = load_base_model(
			model_type=args.enc_model_type,
			config_name=get_huggingface_path(args.enc_model_type),
			model_path=get_huggingface_path(args.enc_model_type),
		)
		self.args = args
		self.config = config
		self.base = base  # CodeBERT model
		self.freeze_base = freeze_base
		
		# # Base model does not require any training - freeze the weights
		if self.freeze_base:
			for param in self.base.parameters():
				param.requires_grad = False
		
		# For each virtual token, predict a word embedding - mu and logvar
		self.n_virtual_tokens = n_virtual_tokens
		self.word_embedding_dim = word_embedding_dim
		
		# Set params
		dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config, 'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		# Define the head for encoding the row vectors - weighs virtual tokens
		self.row_dropout = torch.nn.Dropout(dropout_prob)
		self.row_dense = torch.nn.Linear(hidden_dim, hidden_dim)
		self.row_out_proj = torch.nn.Linear(hidden_dim, n_virtual_tokens)
		
		# Define the head for encoding the column vectors - weighs the word embedding dimensions
		self.col_dropout = torch.nn.Dropout(dropout_prob)
		self.col_dense = torch.nn.Linear(hidden_dim, hidden_dim)
		self.col_out_proj = torch.nn.Linear(hidden_dim, word_embedding_dim)
		
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
		if self.args.enc_model_type == 'codebert-base':
			# [IMP] Codebert Base is based on RoBERTa model: Getting the seq representation should match with that of
			# default RobertaForSequenceClassification & RobertaClassificationHead
			x = self.base(input_ids, attention_mask=attention_mask)
			x = x[0][:, 0, :]
		
		elif self.args.enc_model_type == 'codet5p-110m-embedding':
			x = self.base(input_ids, attention_mask=attention_mask)
		
		else:
			# Should be used for decoder models
			x = self.base(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=input_ids,
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
		
		# Predict the column weights
		col_weights = self.col_dropout(x)
		col_weights = self.col_dense(col_weights)
		col_weights = torch.nn.functional.tanh(col_weights)
		col_weights = self.col_out_proj(col_weights)
		
		# Multiply: uk ∈ R^l, vk ∈ R^d -> uk * vk^T ∈ R^(l x d)
		prompt_specific_clf_embedding = torch.einsum('bi,bj->bij', row_weights, col_weights)
		
		return prompt_specific_clf_embedding
	
	def __str__(self):
		return f"MyEncoder/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"MyEncoder/{self.args.enc_model_type}"


class IDPGSoftPromptGenerator(torch.nn.Module):
	def __init__(self, args, n_virtual_tokens, word_embedding_dim):
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
		dropout_prob = config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config, 'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		# Define the head for encoding all virtual tokens
		self._dropout = torch.nn.Dropout(dropout_prob)
		self.layer_down_project = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
		self.layer_up_project = torch.nn.Linear(hidden_dim, n_virtual_tokens * word_embedding_dim, bias=True)
		
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
		if self.args.enc_model_type == 'codebert-base':
			x = self.base(input_ids, attention_mask=attention_mask)
			x = x[0][:, 0, :]
		
		elif self.args.enc_model_type == 'codet5p-110m-embedding':
			x = self.base(input_ids, attention_mask=attention_mask)
		
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


