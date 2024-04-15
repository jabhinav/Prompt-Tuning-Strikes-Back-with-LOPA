import json
from typing import Dict, List

import torch
from tqdm import tqdm

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftCvaeModel
from utils.config import get_config
from utils.eval import save_predictions_mbxp_format
from utils.data import MBPP_Dataset_wEnc as CustomDataset
from utils.model import PromptSpecificClarificationEncoder as EmbeddingEncoder
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def load_encoder(args, logger, tokenizer):
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
	model.load_state_dict(loaded_state_dict, strict=True)

	# release memory
	del loaded_state_dict

	# Log the loaded checkpoint
	msg = "[INFO] Loaded encoder checkpoint from path: {}".format(args.clf_predictor_path)
	logger.info(msg)
	print(msg)
	
	return model


def load_decoder(args, logger, tokenizer):
	# Get the decoder model
	_, decoder = load_base_model(
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
		decoder.load_state_dict(loaded_state_dict, strict=True)
		
		# release memory
		del loaded_state_dict
		
		# Log the loaded checkpoint
		message = "[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path)
		logger.info(message)
		print(message)
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.CVAE_CAUSAL_LM,
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
	)
	
	# # Load the model adapters - in place
	decoder = PeftCvaeModel.from_pretrained(
		model=decoder,
		model_id=args.load_adapter_from,  # Must be a directory containing the model files
		config=peft_config,
	)
	
	msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
	logger.info(msg)
	print(msg)
	
	# This should match dimensions of torch.nn.Embedding(total_virtual_tokens, config.token_dim)
	args.total_virtual_tokens = args.num_virtual_tokens * peft_config.num_transformer_submodules
	args.word_embedding_dim = peft_config.token_dim
	
	return decoder


@torch.no_grad()
def evaluate(args, logger):
	# Get the tokenizers
	dec_tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))
	
	# Get the dataset
	dataset = CustomDataset(
		path_to_data=args.path_to_data,
		tokenizer=dec_tokenizer,
		max_prompt_length=args.max_prompt_length,
		max_length=args.max_length,
		sample_problems=args.num_test_problems,
		mode='test',
		enc_tokenizer=enc_tokenizer,
		
		# # Uncomment to use a finer split of the training data to evaluate
		# finer_split=0.50,
		# use_first_half=False
	)
	
	# # Leave this as is to only read prompt for any type of data
	dataset.mode = 'test'
	
	# Load the decoder model
	decoder = load_decoder(args, logger, dec_tokenizer)
	decoder.to(args.device)
	decoder.eval()
	
	# Load the encoder model
	encoder = load_encoder(args, logger, enc_tokenizer)
	encoder.to(args.device)
	encoder.eval()

	# Predict for each sample output by each library
	num_loops = int(args.num_return_sequences / args.num_return_sequences_per_iter)
	oracle_output: Dict[str, List[str]] = {}
	for index in tqdm(
			range(len(dataset)),
			desc="Predicting",
			unit="sample",
			colour="RED",
			position=0,
			leave=False,
			dynamic_ncols=True,
			smoothing=0.04
	):
		sample = dataset.sample(index)
		sample = tuple(tensor.unsqueeze(0).to(args.device) for tensor in sample)
		bert_inputs_ids, bert_mask, input_ids, attention_mask = sample
		
		# Get the latent attention weights
		clf_logits = encoder(input_ids=bert_inputs_ids, attention_mask=bert_mask)
		
		# [Uncomment to] Use Softmax
		# latent_attention_weights = torch.nn.functional.softmax(clf_logits, dim=-1)
		
		# [Uncomment to] Use Sigmoid
		latent_attention_weights = torch.sigmoid(clf_logits)
		
		# Log the attention weights
		logger.info("[{}] Attention weights: {}".format(index, latent_attention_weights[0].detach().cpu().numpy().tolist()))
		
		# If sampling from the logits using Gumbel-Softmax
		# latent_attention_weights = gumbel_softmax(clf_logits, temperature=1.0, eps=1e-20)
		
		# # Set the latent attention weights [Comment to ignore latent weighing]
		decoder.latent_attention_weights = latent_attention_weights
		
		all_responses: List[str] = []
		try:
			
			for _ in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
				top_responses = decoder.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					max_new_tokens=args.max_new_tokens,
					do_sample=args.do_sample,
					num_beams=args.num_beams,
					early_stopping=True if args.num_beams > 1 and not args.do_sample else False,
					temperature=args.temperature if args.do_sample else 1.0,
					top_p=args.top_p if args.do_sample else 1.0,
					num_return_sequences=args.num_return_sequences_per_iter,
				)
				
				top_responses = top_responses.detach().cpu().numpy().tolist()
				if 't5' not in args.model_type:  # For T5, we don't have to remove the prompt
					top_responses = [resp[args.max_prompt_length:] for resp in top_responses]
					top_responses = [dec_tokenizer.decode(resp, skip_special_tokens=False) for resp in top_responses]
					# Split the response at the first occurrence of the end of text token.
					# This works since we append the eos token to responses and make the model predict it
					# Also useful to not consider any text after the first occurrence of the eos token
					top_responses = [resp.split(dec_tokenizer.eos_token)[0] for resp in top_responses]
				else:
					top_responses = [dec_tokenizer.decode(resp, skip_special_tokens=True) for resp in top_responses]
				all_responses.extend(top_responses)
		
		except Exception as e:
			if isinstance(e, UnboundLocalError) and str(
					e) == "local variable 'next_tokens' referenced before assignment":
				# See https://github.com/huggingface/transformers/issues/5118
				logger.exception("Problem text was > specified tokens, so cannot do generation")
				logger.info(e)
				raise e
			else:
				logger.exception("Unexpected exception in generating solution")
				logger.info(e)
				raise e
		
		oracle_output[dataset.ids[index]] = all_responses
	
	# Save the output
	# print(json.dumps(output, indent=4))
	with open(args.save_results_at, 'w') as f:
		json.dump(oracle_output, f, indent=4)
	
	save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')


def main():
	args, logger = get_config()
	
	# # Debug
	# args.do_peft = 1
	# args.load_base_from_path = './logging/codegen-350m/Baseline_1.0/output/pytorch_model.bin'
	# args.load_adapter_from = './logging/20240314-201300/PromptTuningMultiModel'
	# args.clf_predictor_path = './logging/20240314-201300/clf_predictor.pt'
	
	evaluate(args, logger)


if __name__ == '__main__':
	main()

