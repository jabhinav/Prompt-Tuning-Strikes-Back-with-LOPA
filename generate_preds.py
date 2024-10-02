import json
import math
import os
from collections import defaultdict

import torch
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter

from utils.config import get_config
from utils.eval import decode_predictions, save_predictions_mbxp_format
from torch.utils.data.dataloader import DataLoader
from utils.data import CruxEval_Dataset_wEnc, MBPP_Dataset_wEnc, NLG_Dataset_wEnc
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path

from transformers import StoppingCriteria
from torch import LongTensor, FloatTensor, eq
from transformers.modeling_utils import load_sharded_checkpoint
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator


def load_encoder(args, logger, accelerator):
	"""
			Initialize the encoder.
	"""
	if args.peft_method == 'idpg':
		model = IDPGSoftPromptGenerator(
			args=args,
			n_virtual_tokens=args.total_virtual_tokens,
			word_embedding_dim=args.word_embedding_dim
		)
	elif args.peft_method == 'lopa':
		model = LatentPromptAttentionGenerator(
			args=args,
			n_virtual_tokens=args.total_virtual_tokens,
			word_embedding_dim=args.word_embedding_dim
		)
	
	else:
		return None
	
	# Load the model state dict on the CPU to avoid an OOM error.
	loaded_state_dict = torch.load(args.clf_predictor_path, map_location="cpu")
	loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
	if args.peft_method == 'idpg':
		loaded_state_dict = {k: v for k, v in loaded_state_dict.items() if 'base' not in k}  # Remove base model weights
		model.load_state_dict(loaded_state_dict, strict=False)  # strict=False allows for partial loading [IDPG-specific]
	else:
		model.load_state_dict(loaded_state_dict, strict=True)
	
	# release memory
	del loaded_state_dict
	
	# Log the loaded checkpoint
	msg = "[INFO] Loaded encoder checkpoint from path: {}".format(args.clf_predictor_path)
	logger.info(msg)
	if accelerator.is_local_main_process:
		print(msg)
	
	return model


def load_pt(args, logger, accelerator, model):
	from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
	
	if not os.path.exists(args.load_adapter_from):
		logger.error("Please specify the correct path to load the model adapters from")
		raise ValueError("Please specify the correct path to load the model adapters from")

	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.CAUSAL_LM,
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
	)
	
	# Load the model adapters - in place
	model = PeftModel.from_pretrained(
		model=model,
		model_id=args.load_adapter_from,  # Must be a directory containing the model files
		config=peft_config,
	)
	msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
	logger.info(msg)
	if accelerator.is_local_main_process:
		print(msg)
		
	return model


def load_lora(args, logger, accelerator, model):
	from peft import PeftModel
	
	if not os.path.exists(args.load_adapter_from):
		logger.error("Please specify the correct path to load the model adapters from")
		raise ValueError("Please specify the correct path to load the model adapters from")
	
	# # Load the model adapters - in place
	model = PeftModel.from_pretrained(
		model=model,
		model_id=args.load_adapter_from,  # Must be a directory containing the model files
	)
	
	# # Link: https://huggingface.co/docs/peft/en/developer_guides/lora
	
	# merge the adapter weights with the base model. doesn’t keep the adapter weights in memory.
	model.merge_and_unload()
	
	# you need to keep a copy of the weights so you can unmerge the adapter later or delete and load different ones
	# model.merge_adapter()
	
	# unmerge the LoRA layers from the base model
	# model.unmerge_adapter()
	
	msg = "[INFO] Loaded the model adapters from: {}".format(args.load_adapter_from)
	logger.info(msg)
	if accelerator.is_local_main_process:
		print(msg)
		
	return model


def load_idpg(args, logger, accelerator, model):
	from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftIDPGModel
	
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


def load_lopa(args, logger, accelerator, model):
	from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, PeftLopaModel
	if not os.path.exists(args.load_adapter_from):
		logger.error("Please specify the correct path to load the model adapters from")
		raise ValueError("Please specify the correct path to load the model adapters from")
	
	# Get the config
	peft_config = PromptTuningConfig(
		task_type=TaskType.CVAE_CAUSAL_LM,
		prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
		num_virtual_tokens=args.num_virtual_tokens,
	)
	
	# # Load the model adapters - in place
	model = PeftLopaModel.from_pretrained(
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


def load_foundation_model(args, logger, accelerator):
	
	# Get the foundation model
	_, model = load_base_model(
		model_type=args.model_type,
		config_name=args.config_name,
		model_path=args.model_name_or_path,
		load_in_8bit=args.load_in_8bit
	)
	
	# [FFT] If the sharded checkpoint directory is provided, load the sharded checkpoint
	if args.sharded_checkpoint_dir is not None:
		# Ref: https://huggingface.co/docs/transformers/big_models
		load_sharded_checkpoint(model, args.sharded_checkpoint_dir)
		msg = "[INFO] Loaded the sharded checkpoint from: {}".format(args.sharded_checkpoint_dir)
		logger.info(msg)
		if accelerator.is_local_main_process:
			print(msg)
	
	# [FFT] If the single checkpoint path is provided, load the checkpoint
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
			
	# [For PEFT methods]
	if args.peft_method == 'pt':
		model = load_pt(args, logger, accelerator, model)
		
	elif args.peft_method == 'lora':
		model = load_lora(args, logger, accelerator, model)
		
	elif args.peft_method == 'idpg':
		model = load_idpg(args, logger, accelerator, model)
		
	elif args.peft_method == 'lopa':
		model = load_lopa(args, logger, accelerator, model)
	
	return model


@torch.no_grad()
def generate(args, logger):
	transformers.logging.set_verbosity_error()
	accelerator = Accelerator()
	
	# ################################ Load the tokenizer ################################################## #
	fm_tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
	enc_tokenizer = load_tokenizer(args.enc_model_type, get_huggingface_path(args.enc_model_type))
	
	# ################################ Prepare the dataset ################################################## #
	if args.task_name == 'mbpp':
		dataset = MBPP_Dataset_wEnc(
			path_to_data=args.path_to_data,
			tokenizer=fm_tokenizer,
			max_prompt_length=args.max_prompt_length,
			max_length=args.max_length,
			mode='test',
			enc_tokenizer=enc_tokenizer,
		)
		
		# Prepare the generation kwargs
		kwargs = {
			"max_new_tokens": args.max_new_tokens,
			"do_sample": args.do_sample,
			"num_beams": args.num_beams,
			"early_stopping": True if args.num_beams > 1 and not args.do_sample else False,
			"temperature": args.temperature if args.do_sample else 1.0,
			"top_p": args.top_p if args.do_sample else 1.0,
			"num_return_sequences": args.num_return_sequences,
		}
		
	elif 'cruxeval' in args.task_name:
		# To match cruxeval implementation
		fm_tokenizer.truncation_side = 'left'
		fm_tokenizer.padding_side = 'right'
		
		# Extract the type of cruxeval task
		assert args.task_name.startswith('cruxeval_')
		cruxeval_task = args.task_name[len('cruxeval_'):]
		dataset = CruxEval_Dataset_wEnc(
			tokenizer=fm_tokenizer,
			max_length=args.max_length,
			mode='test',
			enc_tokenizer=enc_tokenizer,
			cruxeval_task=cruxeval_task,
			prefix=args.prefix,
			cot=args.cot,
		)
		
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
		stop_list.append(fm_tokenizer.eos_token)  # Add eos token to the stop list
		stop_token_ids = [fm_tokenizer(t, return_tensors='pt', add_special_tokens=False)['input_ids'] for t in stop_list]
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
			
	elif args.task_name in ['nlg_e2e', 'nlg_webnlg', 'nlg_dart']:
		dataset = NLG_Dataset_wEnc(
			path_to_data=args.path_to_test_data,
			tokenizer=fm_tokenizer,
			max_length=args.max_length,
			mode='test',
			enc_tokenizer=enc_tokenizer,
			max_eval_length=args.max_new_tokens,
		)
		
		# Prepare the generation kwargs
		kwargs = {
			"max_new_tokens": args.max_new_tokens,
			"do_sample": args.do_sample,
			"num_return_sequences": args.num_return_sequences,
			"num_beams": args.num_beams,
			"length_penalty": args.length_penalty,
			"no_repeat_ngram_size": args.no_repeat_ngram_size,
			"repetition_penalty": args.repetition_penalty,
			"temperature": args.temperature,
		}
		
	else:
		raise ValueError(f"Please specify the correct task name. {args.task_name} is not supported.")
	
	# # Leave this as is to only read prompt for any type of data
	ds_loader = DataLoader(dataset, batch_size=1)
	
	# ################################ Get the model ########################################################## #
	foundation_model = load_foundation_model(args, logger, accelerator)
	foundation_model.eval()
	
	# Get the encoder
	encoder = load_encoder(args, logger, accelerator)
	if encoder is not None:
		encoder.eval()
	
	if args.load_in_8bit:
		# decoder.to() is not supported for 8bit and 4bit models
		if encoder is not None:
			encoder, foundation_model, ds_loader = accelerator.prepare(encoder, foundation_model, ds_loader)
		else:
			foundation_model, ds_loader = accelerator.prepare(foundation_model, ds_loader)
	else:
		# we only wrap data loader to avoid extra memory occupation
		foundation_model = foundation_model.to(accelerator.device)
		if encoder is not None:
			encoder = encoder.to(accelerator.device)
		ds_loader = accelerator.prepare(ds_loader)
		
	
	# ################################ Generate the predictions ################################################ #
	gen_token_dict = defaultdict(list)  # dict of list of generated tokens
	for step, batch in tqdm(
			enumerate(ds_loader),
			total=math.ceil(len(dataset) / accelerator.num_processes),
			desc="Generating Predictions",
			colour="GREEN",
			# leave=False,
			dynamic_ncols=True,
			smoothing=0.04,
			disable=not accelerator.is_main_process,
	):
		
		
		# ################################ Prepare the input ################################################ #
		if 'cruxeval' in args.task_name:
			inputs = batch["input_ids"][:, :batch["input_len"]]  # this removes the padding tokens on the right
			attention_mask = None
			# Since we are loading 1 sample batch per gpu
			num_tokens = len(inputs[0])
			
			# Skip the samples for which number of tokens in the input exceeds the max_length
			if num_tokens >= args.max_length:
				logger.info(
					f"Skipping task {batch['task_id'][0]} as the input length ={num_tokens} exceeds max_length ={args.max_length}")
				gen_token_dict[batch["task_id"][0]].extend([[]] * args.num_return_sequences)
				continue
			
			# Update max new tokens to amount of tokens left
			kwargs["max_new_tokens"] = args.max_length - num_tokens
			kwargs["stopping_criteria"] = [StopOnTokens()]  # create a new instance for each batch
		
		elif args.task_name in ['nlg_e2e', 'nlg_webnlg', 'nlg_dart']:
			inputs = batch["input_ids"][:, :batch["input_len"]]  # this removes the padding tokens on the right
			attention_mask = batch["attention_mask"][:, :batch["input_len"]]

		else:
			inputs = batch["input_ids"]
			attention_mask = batch["attention_mask"]
			
		# ################################ Get the encoder output ################################################ #
		if args.peft_method == 'idpg':
			inst_sp_soft_prompt = encoder(input_ids=batch["enc_input_ids"], attention_mask=batch["enc_attention_mask"])
			foundation_model.soft_prompt = inst_sp_soft_prompt
			
		elif args.peft_method == 'lopa':
			inst_sp_soft_prompt = encoder(input_ids=batch["enc_input_ids"], attention_mask=batch["enc_attention_mask"])
			# Gating mechanism
			soft_prompt_attention_weights = torch.sigmoid(inst_sp_soft_prompt)
			foundation_model.latent_attention_weights = soft_prompt_attention_weights

		# ############################################ Predict ################################################ #
		is_wrapped = args.load_in_8bit
		if is_wrapped:
			# 8bit and 4bit models are wrapped in accelerator
			generated_tokens = accelerator.unwrap_model(foundation_model).generate(
				input_ids=inputs,
				attention_mask=attention_mask,
				**kwargs
			)
		else:
			generated_tokens = foundation_model.generate(
				input_ids=inputs,
				attention_mask=attention_mask,
				**kwargs
			)
			
		# each task is generated batch_size times
		generated_tasks = batch["task_id"].repeat(args.num_return_sequences)
		generated_tokens = accelerator.pad_across_processes(
			generated_tokens, dim=1, pad_index=fm_tokenizer.pad_token_id
		)
		generated_tokens, generated_tasks = accelerator.gather(
			(generated_tokens, generated_tasks)
		)
		generated_tokens = generated_tokens.cpu().numpy()
		generated_tasks = generated_tasks.cpu().numpy().tolist()
		
		for task_idx, generated_tokens in zip(generated_tasks, generated_tokens):
			gen_token_dict[task_idx].append(generated_tokens)
	
	# ################################ Decode the predictions ############################################### #
	decoded_preds_processed, decoded_preds_raw = decode_predictions(args, gen_token_dict, fm_tokenizer, dataset)
	
	# ################################ Save the predictions ################################################## #
	if 'cruxeval' in args.task_name:
		references = {
			dataset.idx_to_id[row_idx]: dataset.task.get_reference(dataset.data[dataset.idx_to_pos[row_idx]])
			for row_idx in gen_token_dict.keys()
		}
		
		if accelerator.is_main_process:
			with open(os.path.join(args.log_dir, 'output.json'), 'w') as f:
				json.dump(decoded_preds_processed, f, indent=4)
			logger.info(f"Saved the output in {args.log_dir}")
			
			with open(os.path.join(args.log_dir, 'output_raw.json'), 'w') as f:
				json.dump(decoded_preds_raw, f, indent=4)
			logger.info(f"Saved the raw output in {args.log_dir}")
			
			with open(os.path.join(args.log_dir, 'references.json'), 'w') as f:
				json.dump(references, f, indent=4)
			logger.info(f"Saved the references in {args.log_dir}")
		
	elif args.task_name == 'mbpp':
		
		if accelerator.is_main_process:
			# Map back the task ids to the original ids
			oracle_output = {dataset.ids[k]: v for k, v in enumerate(decoded_preds_processed)}
			
			# Save the output
			with open(args.save_results_at, 'w') as f:
				json.dump(oracle_output, f, indent=4)
			
			save_predictions_mbxp_format(args, oracle_output, lang='python', d_type='MBPP')
	
	elif args.task_name == 'nlg_e2e':
		if accelerator.is_main_process:
			with open(os.path.join(args.log_dir, 'e2e_ref.txt'), 'w', encoding='utf8') as ref_writer:
				with open(os.path.join(args.log_dir, 'e2e_pred.txt'), 'w', encoding='utf8') as pred_writer:
					with open(os.path.join(args.log_dir, 'e2e_raw.txt'), 'w', encoding='utf8') as raw_writer:
						with open(os.path.join(args.log_dir, 'e2e_context.txt'), 'w', encoding='utf8') as prompt_writer:
							for context in decoded_preds_processed:
								gen_text = decoded_preds_processed[context]['generated_text']
								references = decoded_preds_processed[context]['references']
								hypothesis = decoded_preds_processed[context]['hypothesis']
								for ref in references:
									ref_writer.write(ref + '\n')
								ref_writer.write('\n')
								pred_writer.write(hypothesis + '\n')
								raw_writer.write(gen_text + '\n')
								prompt_writer.write(context + '\n')
	
	elif args.task_name in ['nlg_webnlg', 'nlg_dart']:
		if accelerator.is_main_process:
			categories = ['seen', 'unseen', 'all']
			
			for cate in categories:
				result_dir = os.path.join(args.log_dir, cate)
				if not os.path.exists(result_dir):
					os.makedirs(result_dir)
			
				output_ref_file = os.path.join(result_dir, 'references_{}'.format(args.task_name.split('_')[1]))
				output_pred_file = os.path.join(result_dir, 'hypothesis_{}'.format(args.task_name.split('_')[1]))
				
				with open(output_pred_file, 'w', encoding='utf8') as pred_writer:
					with open(os.path.join(result_dir, 'raw.txt'), 'w', encoding='utf8') as raw_writer:
						with open(os.path.join(result_dir, 'context.txt'), 'w', encoding='utf8') as prompt_writer:
							if not os.path.exists(output_ref_file):
								os.makedirs(output_ref_file)
							
							ref_num = 6  # Both for webnlg and dart
							reference_writers = [
								open(os.path.join(output_ref_file, f'reference{fid}'), 'w', encoding='utf8')
								for fid in range(0, ref_num)
							]
							for context in decoded_preds_processed:
								
								if cate == 'seen' and not decoded_preds_processed[context]['cate']:
									# Skip samples which are not seen
									continue
								if cate == 'unseen' and decoded_preds_processed[context]['cate']:
									# Skip samples which are seen
									continue
								
								gen_text = decoded_preds_processed[context]['generated_text']
								references = decoded_preds_processed[context]['references']
								hypothesis = decoded_preds_processed[context]['hypothesis']
								
								# This loop will always write 6 references (will repeat in case of less references)
								for fid in range(0, ref_num):
									if len(references) > fid:
										reference_writers[fid].write(references[fid] + '\n')
									else:
										reference_writers[fid].write(references[0] + '\n')
								
								pred_writer.write(hypothesis + '\n')
								raw_writer.write(gen_text + '\n')
								prompt_writer.write(context + '\n')
							
							# Close the reference writers
							for writer in reference_writers:
								writer.close()

	else:
		raise ValueError(f"Please specify the correct task name. {args.task_name} is not supported.")


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	# # Debug
	# args.do_peft = 1
	# args.load_base_from_path = './logging/Baseline_0.50/output/pytorch_model.bin'
	# args.load_adapter_from = './logging/e2e_pt_m100/final/PEFT'
	# args.clf_predictor_path = './logging/e2e_pt_m100/final/clf_predictor.pt'
	
	generate(args, logger)


if __name__ == '__main__':
	main()