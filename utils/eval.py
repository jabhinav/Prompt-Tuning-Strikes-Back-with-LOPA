import torch
from utils.model import get_response_log_probs_for_lib, compute_responsibilities_with_prior


@torch.no_grad()
def compute_Q_func(args, batch, tokenizer, model, prior=None):
	
	bert_prompt, bert_prompt_mask, prompt, prompt_mask, response, response_mask = batch

	responsibilities = compute_responsibilities_with_prior(args, batch, tokenizer, model, prior)
	responsibilities = responsibilities.clamp(min=1e-8)
	responsibilities.to(args.device)
	
	# Compute Q-function: Total log-likelihood of the data coming from library, metric to track convergence
	# Init. with the prior component of the Q-func
	if prior is None:
		# Compute the prior probabilities (uniform)
		clf_preds = torch.ones_like(responsibilities) / args.num_libraries
		clf_preds.to(args.device)
	else:
		clf_preds = prior(bert_prompt, bert_prompt_mask)
	
	q_func = (responsibilities * torch.log(clf_preds + 1e-8)).sum(dim=-1).mean().detach().cpu().numpy().item()
	
	for k in range(args.num_libraries):
		# Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
		resp_log_prob = get_response_log_probs_for_lib(
			args,
			(prompt, prompt_mask, response, response_mask),
			tokenizer,
			model,
			k
		)
		q_func += (resp_log_prob * responsibilities[:, k]).mean().detach().cpu().numpy().item()
		
	return q_func

