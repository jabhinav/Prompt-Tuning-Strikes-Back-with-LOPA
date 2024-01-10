from typing import Tuple

import torch
from torch.nn import functional as F

from utils.xformer import load_base_model
from utils.custom import is_rank_0


class ClarificationMLPPredictor(torch.nn.Module):
	def __init__(self, input_dim: int, output_dim: int):
		super(ClarificationMLPPredictor, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.fc1 = torch.nn.Linear(self.input_dim, 1024)
		self.fc2 = torch.nn.Linear(1024, 512)
		self.fc3 = torch.nn.Linear(512, 256)
		self.fc4 = torch.nn.Linear(256, self.output_dim)
	
	def forward(self, prompt_embedding: torch.Tensor) -> torch.Tensor:
		"""
		:param prompt_embedding: Tensor of shape (batch_size, input_dim)
		:return: Tensor of shape (batch_size, output_dim)
		"""
		x = torch.nn.functional.relu(self.fc1(prompt_embedding))
		x = torch.nn.functional.relu(self.fc2(x))
		x = torch.nn.functional.relu(self.fc3(x))
		x = self.fc4(x)  # Logits
		x = torch.nn.functional.softmax(x, dim=1)  # Probabilities
		return x
	
	def predict(self, prompt_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param prompt_embedding: Tensor of shape (batch_size, input_dim)
		:return: Tuple of (probabilities, indices)
		"""
		probabilities = self.forward(prompt_embedding)
		indices = torch.argmax(probabilities, dim=1)
		return probabilities, indices


class ClarificationCodeBERTPredictor(torch.nn.Module):
	def __init__(self, args, output_dim):
		super(ClarificationCodeBERTPredictor, self).__init__()
		
		config, base = load_base_model(
			model_type=args.bert_model_type,
			config_name=args.bert_config_name,
			model_path=args.bert_model_name_or_path,
		)
		self.config = config
		self.base = base  # CodeBERT model
		
		# Classifier head
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
		self.out_proj = torch.nn.Linear(config.hidden_size, output_dim)
		self.init_predictor_head()
	
	def init_predictor_head(self):
		# Initialize the weights
		self.dense.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		self.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		self.out_proj.bias.data.zero_()
	
	def forward(self, input_ids, attention_mask):
		x = self.base(input_ids, attention_mask=attention_mask)
		x = x[0][:, 0, :]  # Get the CLS token embedding
		# x = x.pooler_output  # Get the pooled output
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.nn.functional.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x
	
	# def predict(self, input_ids, attention_mask):
	# 	if self.return_logits:
	# 		logits = self.forward(input_ids, attention_mask)
	# 		probabilities = torch.nn.functional.softmax(logits, dim=-1)
	# 	else:
	# 		probabilities = self.forward(input_ids, attention_mask)
	# 	indices = torch.argmax(probabilities, dim=1)
	# 	return probabilities, indices


def logprobs_from_logits(logits, labels):
	"""
	See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
	"""
	log_p = F.log_softmax(logits, dim=2)
	logpy = torch.gather(log_p, 2, labels.unsqueeze(2)).squeeze(-1)
	return logpy


def get_response_log_probs(args, batch, tokenizer, model, library_idx):
	prompt, prompt_mask, response, response_mask = batch
	batch_size = prompt.size(0)
	
	# # Set the library index
	# print("[Debug] Library Index:", library_idx)
	# model.library_idx = library_idx
	# print("[Debug] Model Library Index:", model.library_idx)
	
	resp_logits = model(
		library_idx=library_idx,
		input_ids=prompt,
		attention_mask=prompt_mask,
		labels=response,
		output_hidden_states=True
	)['logits']
	
	# # Prepare the response mask for the complete response (including for the latent prompt)
	# Append response_mask with 0s for the latent prompt (this is not the mask for attending to latent prompt)
	response_prefix_mask = torch.zeros((batch_size, args.num_virtual_tokens)).to(response_mask.device)
	response_mask = torch.cat((response_prefix_mask, response_mask), dim=1)
	
	# # Prepare labels for the complete response (including for the latent prompt)
	# Append labels [=-100] for the latent prompt to the response
	response_prefix = torch.full((batch_size, args.num_virtual_tokens), -100).to(response.device)
	response = torch.cat((response_prefix, response), dim=1)
	response[response == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
	resp_labels = response.contiguous()
	
	# # Compute the log-probability of the response tokens
	resp_log_prob = logprobs_from_logits(resp_logits, resp_labels)
	resp_log_prob = resp_log_prob * response_mask
	
	# Likelihood of the sample coming from the latent prompt of library k
	resp_log_prob = resp_log_prob.sum(dim=1)
	
	return resp_log_prob


@torch.no_grad()
def compute_responsibilities(args, batch, tokenizer, model, prior: ClarificationCodeBERTPredictor = None) -> torch.Tensor:
	"""
	Compute the responsibilities i.e. posterior probabilities of the sample coming from the latent prompt of each
	library.
	:param args:
	:param batch: (prompt_embed, prompt, prompt_mask, response, response_mask)
	:param tokenizer:
	:param model:
	:param prior:
	:return:
	"""
	
	batch_size = batch[0].size(0)
	if len(batch) == 6:  # Do not use the prior as indicative of whether additional prompt is present
		prior_prompt, prior_prompt_mask, prompt, prompt_mask, response, response_mask = batch
	else:
		prompt, prompt_mask, response, response_mask = batch
		prior_prompt, prior_prompt_mask = None, None
	
	# Create a tensor of shape (n_samples, num_libraries) to store the responsibilities
	likelihood = torch.zeros((batch_size, args.num_libraries)).to(args.device)
	
	for k in range(args.num_libraries):
		# Store the likelihood of the sample coming from the latent prompt of library k
		likelihood[:, k] = get_response_log_probs(
			args,
			(prompt, prompt_mask, response, response_mask),
			tokenizer,
			model,
			k
		)
	
	if prior is None:
		# Compute the responsibilities (prior is uniform, thus cancelled out)
		responsibilities = F.softmax(likelihood, dim=1)
	else:
		# Compute the prior probabilities
		clf_logits = prior(prior_prompt, prior_prompt_mask)
		log_prior = torch.nn.functional.log_softmax(clf_logits, dim=1)
		log_prior = log_prior.to(args.device)
		
		# Add the log_prior to the likelihood
		full_likelihood = likelihood + log_prior
		
		# Compute the responsibilities
		responsibilities = F.softmax(full_likelihood, dim=1)
	
	# To prevent underflow, clip the responsibilities to a minimum value
	responsibilities = responsibilities.clamp(min=1e-8)
	
	responsibilities = responsibilities.detach()
	
	responsibilities.to(args.device)
	return responsibilities


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
