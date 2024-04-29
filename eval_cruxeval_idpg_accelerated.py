import json
import math
import os
from collections import defaultdict

import torch
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftIDPGModel
from utils.config import get_config
from utils.eval import decode_cruxeval_predictions
from torch.utils.data.dataloader import DataLoader
from utils.data import CruxEval_Dataset_wEnc as CustomDataset
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path
from utils.model import LatentPromptAttentionGenerator as EmbeddingEncoder

from transformers import StoppingCriteria
from torch import LongTensor, FloatTensor, eq


def load_encoder(args, logger, accelerator):
	"""
			Initialize the encoder.
	"""
	
	model = EmbeddingEncoder(
		args=args,
		n_virtual_tokens=args.total_virtual_tokens,
		word_embedding_dim=args.word_embedding_dim
	)
	
	# Load the model state dict on the CPU to avoid an OOM error.
	loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
	loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
	loaded_state_dict = {k: v for k, v in loaded_state_dict.items() if 'base' not in k}  # Remove base model weights
	model.load_state_dict(loaded_state_dict, strict=False)  # strict=False allows for partial loading [IDPG-specific]
	
	# release memory
	del loaded_state_dict
	
	# Log the loaded checkpoint
	msg = "[INFO] Loaded encoder checkpoint from path: {}".format(args.clf_predictor_path)
	logger.info(msg)
	if accelerator.is_local_main_process:
		print(msg)
	
	return model


def load_decoder(args, logger, accelerator):
	# Get the model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# Load checkpoint
	if args.load_base_from_path is not None:
		# We load the model state dict on the CPU to avoid an OOM error.
		loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		message = "[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		logger.info(message)
		if accelerator.is_local_main_process:
			print(message)
	
	if not os.path.exists(args.load_adapter_from):
		logger.error("Please specify the correct path to load the model adapters from")
		raise ValueError("Please specify the correct path to load the model adapters from")
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.IDPG_CAUSAL_LM,
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
	)
	
	# # Load the model adapters - in place
	model = PeftIDPGModel.from_pretrained(
		model=model,
		model_id=args.load_adapter_from,  # Must be a directory containing the model files
		config=peft_config,
	)
	
	msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
	logger.info(msg)
	if accelerator.is_local_main_process:
		print(msg)
	
	# This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
	args.total_virtual_tokens = args.num_virtual_tokens * peft_config.num_transformer_submodules
	args.word_embedding_dim = peft_config.token_dim
	
	return model


@torch.no_grad()
def evaluate(args, logger):
	transformers.logging.set_verbosity_error()
	accelerator = Accelerator()
	
	# Get the tokenizer
	tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))
	
	# To match cruxeval implementation
	tokenizer.truncation_side = 'left'
	tokenizer.padding_side = 'right'
	
	# Get the dataset
	dataset = CustomDataset(
		cruxeval_task=args.cruxeval_task,
		tokenizer=tokenizer,
		max_length=args.max_length,
		prefix=args.prefix,
		cot=args.cot,
		enc_tokenizer=enc_tokenizer,
		mode='test'
	)
	
	# # Leave this as is to only read prompt for any type of data
	ds_loader = DataLoader(dataset, batch_size=1)
	
	# Get the decoder
	decoder = load_decoder(args, logger, accelerator)
	decoder.eval()
	
	# Get the encoder
	encoder = load_encoder(args, logger, accelerator)
	encoder.eval()
	
	if args.load_in_8bit:
		# decoder.to() is not supported for 8bit and 4bit models
		encoder, decoder, ds_loader = accelerator.prepare(encoder, decoder, ds_loader)
	else:
		# we only wrap data loader to avoid extra memory occupation
		decoder = decoder.to(accelerator.device)
		encoder = encoder.to(accelerator.device)
		ds_loader = accelerator.prepare(ds_loader)
	
	# Predict for each sample output by each library
	# oracle_output: Dict[str, Dict[str, List[str]]] = {}
	
	# Prepare the generation kwargs
	kwargs = {
		"max_new_tokens": args.max_length,
		"do_sample": args.do_sample,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"num_return_sequences": args.num_return_sequences,
	}
	
	# Let's define the stopping criteria
	stop_list = dataset.task.stop_words
	stop_list.append(tokenizer.eos_token)  # Add eos token to the stop list
	stop_token_ids = [tokenizer(t, return_tensors='pt', add_special_tokens=False)['input_ids'] for t in stop_list]
	stop_token_ids = [LongTensor(t).to(accelerator.device) for t in stop_token_ids]
	
	class StopOnTokens(StoppingCriteria):
		def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
			# Check if the stop tokens are present in the generated tokens
			for stop_token_id in stop_token_ids:
				# Don't know why 1: since we are not adding the special token while tokenizing
				# if (input_ids[0][-len(stop_token_id[0])+1:] == stop_token_id[0][1:]).all():
				if (input_ids[0][-len(stop_token_id[0]):] == stop_token_id[0]).all():
					return True
			return False
	
	gen_token_dict = defaultdict(list)  # dict of list of generated tokens
	for step, batch in tqdm(
			enumerate(ds_loader),
			total=math.ceil(len(dataset) / accelerator.num_processes),
			desc=f"Generating Predictions (for {args.cruxeval_task})",
			colour="GREEN",
			# leave=False,
			dynamic_ncols=True,
			smoothing=0.04,
			disable=not accelerator.is_main_process,
	):
		
		inputs = batch["input_ids"][:, :batch["input_len"]]  # this removes the padding tokens on the right
		# Since we are loading 1 sample batch per gpu
		num_tokens = len(inputs[0])
		
		# Get the encoder
		soft_prompt = encoder(input_ids=batch["enc_input_ids"], attention_mask=batch["enc_attention_mask"])
		decoder.soft_prompt = soft_prompt
		
		# Skip the samples for which number of tokens in the input exceeds the max_length
		if num_tokens >= args.max_length:
			logger.info(
				f"Skipping task {batch['task_id'][0]} as the input length ={num_tokens} exceeds max_length ={args.max_length}")
			gen_token_dict[batch["task_id"][0]].extend([[]] * args.num_return_sequences)
			continue
		
		# Update max new tokens to amount of tokens left
		kwargs["max_new_tokens"] = args.max_length - num_tokens
		kwargs["stopping_criteria"] = [StopOnTokens()]  # create a new instance for each batch
		
		is_wrapped = args.load_in_8bit
		if is_wrapped:
			# 8bit and 4bit models are wrapped in accelerator
			generated_tokens = accelerator.unwrap_model(decoder).generate(
				input_ids=inputs,
				**kwargs
			)
		else:
			generated_tokens = decoder.generate(
				input_ids=inputs,
				**kwargs
			)
		
		# each task is generated batch_size times
		generated_tasks = batch["task_id"].repeat(args.num_return_sequences)
		generated_tokens = accelerator.pad_across_processes(
			generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
		)
		generated_tokens, generated_tasks = accelerator.gather(
			(generated_tokens, generated_tasks)
		)
		generated_tokens = generated_tokens.cpu().numpy()
		generated_tasks = generated_tasks.cpu().numpy().tolist()
		
		for task_idx, generated_tokens in zip(generated_tasks, generated_tokens):
			gen_token_dict[task_idx].append(generated_tokens)
	
	code_gens, code_gens_raw = decode_cruxeval_predictions(gen_token_dict, tokenizer, dataset)
	references = {
		dataset.idx_to_id[row_idx]: dataset.task.get_reference(dataset.data[dataset.idx_to_pos[row_idx]])
		for row_idx in gen_token_dict.keys()
	}
	
	if accelerator.is_main_process:
		with open(os.path.join(args.log_dir, 'output.json'), 'w') as f:
			json.dump(code_gens, f, indent=4)
		logger.info(f"Saved the output in {args.log_dir}")
		
		with open(os.path.join(args.log_dir, 'output_raw.json'), 'w') as f:
			json.dump(code_gens_raw, f, indent=4)
		logger.info(f"Saved the raw output in {args.log_dir}")
		
		with open(os.path.join(args.log_dir, 'references.json'), 'w') as f:
			json.dump(references, f, indent=4)
		logger.info(f"Saved the references in {args.log_dir}")


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	evaluate(args, logger)


if __name__ == '__main__':
	main()
