import json
import os
from collections import OrderedDict

import wandb
import torch
from matplotlib import pyplot as plt
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm

from utils.model import ClarificationCodeBERTPredictor
from utils.config import get_config
from utils.xformer import load_tokenizer, get_huggingface_path
from utils.data import MBPP_Dataset_only_CodeBERT as CustomDataset
from utils.custom import is_rank_0


def use_prior_to_get_clf_idxs(args, logger):
	from eval import get_clf_idx_using_NN
	from utils.data import MBPP_Dataset_w_CodeBERT as Dataset
	from typing import Dict, List
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	
	# Get the dataset
	dataset = Dataset(
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test'
	)
	
	# Uncomment to use when testing on training data
	# dataset.mode = 'test'
	
	get_clf_idx = get_clf_idx_using_NN(args, logger)  # Define the function to get the library index
	
	most_likely_lib_idx: Dict[str, List[int]] = {}
	for index in tqdm(range(len(dataset)), desc="Predicting", position=0, leave=True):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		bert_inputs_ids, bert_mask, _, _ = sample
		
		# ############################## Add logic for identifying the library index ############################## #
		# chosen_lib_idx = get_clf_idx_using_prompt_ll(args, input_ids, attention_mask, tokenizer, model)
		# chosen_lib_idx = get_clf_idx(input_embeds)  # For MLP as NN classifier
		chosen_clf_idx = get_clf_idx(bert_inputs_ids, bert_mask)  # For BERT-like model as NN classifier
		chosen_clf_idx = int(chosen_clf_idx)
		
		most_likely_lib_idx[dataset.ids[index]] = [int(chosen_clf_idx)]
	
	# Save the output as json file
	with open(os.path.join(args.log_dir, f'most_likely_lib_idx_using_prior_net.json'), 'w') as file:
		json.dump(most_likely_lib_idx, file)


@torch.no_grad()
def eval_clf_predictor(args, logger, epoch, model, dataset, mode):
	pred_labels, pred_probs = {}, {}
	clf_count = OrderedDict()
	model.eval()
	for idx in tqdm(range(len(dataset)), desc=f"Predicting {mode} Labels", position=0, leave=True):
		sample = dataset.sample(idx)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		
		# Get the prediction
		logits = model(sample[0], sample[1])
		probabilities = torch.nn.functional.softmax(logits, dim=-1)
		indices = torch.argmax(probabilities, dim=1)
		pred_probs[dataset.ids[idx]] = probabilities[0].detach().cpu().numpy().tolist()
		pred_labels[dataset.ids[idx]] = indices.detach().cpu().item()
		
		# Count the number of times each library is predicted
		clf_count[indices.detach().cpu().item()] = clf_count.get(indices.detach().cpu().item(), 0) + 1
	
	# Sort the count
	clf_count = OrderedDict(sorted(clf_count.items(), key=lambda x: x[1], reverse=True))
	
	# Save the predictions
	with open(os.path.join(args.log_dir, f'{mode}_lib_predictions_{epoch}.json'), 'w') as file:
		json.dump(pred_labels, file, indent=4)
	# Save the probabilities
	with open(os.path.join(args.log_dir, f'{mode}_lib_probabilities_{epoch}.json'), 'w') as file:
		json.dump(pred_probs, file, indent=4)
		
	# Save the count
	logger.info(f"Epoch {epoch} pred_clf_count ({mode}): {clf_count}")
	print(f"Epoch {epoch} pred_clf_count ({mode}): {clf_count}")
	
	# If G.T. labels are available, compute the accuracy
	if dataset.labels:
		correct = 0
		for idx in range(len(dataset)):
			if isinstance(dataset.labels[dataset.ids[idx]], list):
				gt_label = dataset.labels[dataset.ids[idx]][0]
			else:
				gt_label = dataset.labels[dataset.ids[idx]]
			if pred_labels[dataset.ids[idx]] == gt_label:
				correct += 1
		accuracy = correct / len(dataset)
		logger.info(f"Epoch {epoch} accuracy ({mode}): {accuracy}")
		print(f"Epoch {epoch} accuracy ({mode}): {accuracy}")
		
		if args.wandb_logging:
			wandb.log({f"{mode}_accuracy": accuracy})
	
	
def train_clf_predictor(args, logger):
	
	if is_rank_0():
		print(f"\n\nStarting training!! (Using train data split {args.finer_train_split}, First Half: {args.use_train_first_half})\n\n")
	
	# Setup wandb
	if args.wandb_logging:
		wandb.init(project=args.project_name, config=args)
		wandb.run.name = 'CodeBERT-ClfPredictor'
	
	# Add BERT specific args
	args.bert_model_type = "codebert-base"
	args.bert_tokenizer_name = get_huggingface_path(args.bert_model_type)
	args.bert_model_name_or_path = get_huggingface_path(args.bert_model_type)
	args.bert_config_name = get_huggingface_path(args.bert_model_type)
	
	# Load Dataset
	tokenizer = load_tokenizer(args.bert_model_type, args.bert_tokenizer_name)
	dataset = CustomDataset(
		num_classes=args.num_libraries,
		path_to_data=args.path_to_data,
		path_to_labels=args.path_to_train_prior_labels,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_train_problems,
		mode='train',
		
		# Uncomment to use a finer split of the training data to learn the prior
		finer_split=args.finer_train_split,
		use_first_half=args.use_train_first_half
	)
	sampler = RandomSampler(dataset)
	
	test_dataset = CustomDataset(
		num_classes=args.num_libraries,
		path_to_data=args.path_to_data,
		tokenizer=tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test'
	)
	
	args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0, pin_memory=False)
	args.num_training_steps = (len(train_dataloader) * args.num_epochs)
	
	# Get the model
	model = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_libraries)
	
	# Load the model from checkpoint
	if args.clf_predictor_path is not None:
		loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		del loaded_state_dict
	
	# Define the optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.prior_lr)
	
	# GPU-ize the model
	model.to(args.device)
	
	# If number of CUDA devices > 1, use DataParallel
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)
		
	# # Evaluate the model [Debug]
	eval_clf_predictor(args, logger, -1, model, test_dataset, 'test')
	eval_clf_predictor(args, logger, -1, model, dataset, 'train')
	
	# Train the model
	losses = []
	for ep in range(args.num_epochs):
		epoch_loss = 0
		model.train()
		for _ in tqdm(range(len(train_dataloader)), desc=f"Epoch {ep}", position=0, leave=True):
			# Get the batch
			batch = next(iter(train_dataloader))
			batch = tuple(t.to(args.device) for t in batch)
			
			prompt, mask, labels, batched_weights = batch
			
			# Forward pass
			predicted_logits = model(prompt, mask)
			
			# # Let's compute the loss
			# loss = - batched_weights * labels * torch.nn.functional.log_softmax(predicted_logits, dim=-1)
			# loss = loss.sum(dim=-1).mean()
			
			labels = torch.argmax(labels, dim=-1)
			loss = torch.nn.functional.cross_entropy(predicted_logits, labels, weight=dataset.cls_weights.to(args.device))
			
			if args.wandb_logging:
				wandb.log({"loss": loss})
			
			# Backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.detach().cpu().item()
			losses.append(loss.detach().cpu().item())
		
		# Log the epoch loss
		epoch_loss /= len(train_dataloader)
		logger.info(f"Epoch {ep} loss: {epoch_loss}")
		
		# Evaluate the model
		eval_clf_predictor(args, logger, ep, model, dataset, mode='train')
		eval_clf_predictor(args, logger, ep, model, test_dataset, mode='test')
	
	# Plot the losses. X-axis is the number of iterations, Y-axis is the loss
	plt.plot(losses)
	plt.savefig(os.path.join(args.log_dir, f'lib_predictor_losses.png'))
	
	# Unwrap the model from DataParallel
	model = model.module if hasattr(model, 'module') else model
	
	# Save the model
	torch.save(model.state_dict(), os.path.join(args.log_dir, f'lib_predictor.pt'))


def main():
	args, logger = get_config()
	args.per_gpu_train_batch_size = 2
	args.num_epochs = 50
	
	# # # For Debugging
	# args.clf_predictor_path = './logging/clf_predictor.pt'
	# use_prior_to_get_clf_idxs(args, logger)
	
	train_clf_predictor(args, logger)


if __name__ == '__main__':
	main()