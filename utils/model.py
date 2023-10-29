import torch
from torch.nn import functional as F


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
def compute_responsibilities(args, batch, tokenizer, model) -> torch.Tensor:
	"""
	Compute the responsibilities i.e. posterior probabilities of the sample coming from the latent prompt of each
	library.
	:param args:
	:param batch:
	:param tokenizer:
	:param model:
	:return:
	"""
	
	batch_size = batch[0].size(0)
	
	# Create a tensor of shape (n_samples, num_libraries) to store the responsibilities
	likelihood = torch.zeros((batch_size, args.num_libraries)).to(args.device)
	
	for k in range(args.num_libraries):
		# Store the likelihood of the sample coming from the latent prompt of library k
		likelihood[:, k] = get_response_log_probs(args, batch, tokenizer, model, k)
	
	# Normalize the responsibilities (prior can be uniform, thus cancelled out)
	responsibilities = F.softmax(likelihood, dim=1)
	
	return responsibilities.detach()


def compute_grad_norm(model):
	total_norm = 0
	for p in model.parameters():
		if p.grad is not None and p.requires_grad:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
	total_norm = total_norm ** (1. / 2)
	return total_norm
