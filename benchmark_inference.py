import argparse
import copy
from custom_benchmark.benchmark import PyTorchBenchmark
from custom_benchmark.benchmark_args import PyTorchBenchmarkArguments
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator


# _args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[1], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(_args)
# results = benchmark.run()
# print(results)

word_embedding_dims = [2048]  # 1024, 2048, 2560, 3072, 4096
ranks = [1]
token_lens = [5, 10, 25, 50, 100]
for r in ranks:
	for t in token_lens:
		for d in word_embedding_dims:
			print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
			print(f"==================== Benchmarking for d/r/t = {d}/{r}/{t} ====================")
			print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
			parser = argparse.ArgumentParser()
			parser.add_argument('--lp_rank', type=int, default=r)
			parser.add_argument('--n_virtual_tokens', type=int, default=t)
			parser.add_argument('--word_embedding_dim', type=int, default=d)
			args = parser.parse_args()
			
			args1 = copy.deepcopy(args)
			args1.enc_model_type = "codebert-base"
			enc1 = LatentPromptAttentionGenerator(args1, args1.n_virtual_tokens, args1.word_embedding_dim)
			
			args2 = copy.deepcopy(args)
			args2.enc_model_type = 'codebert-base'
			enc2 = IDPGSoftPromptGenerator(args2, args2.n_virtual_tokens, args2.word_embedding_dim)
			
			enc1_name = str(enc1)
			enc2_name = str(enc2)
			# Get number of trainable parameters
			print(f"Number of trainable parameters in Prompt Tuning: {args1.word_embedding_dim * args1.n_virtual_tokens / 1e6:.2f}M")
			trainable_params = sum(p.numel() for p in enc1.parameters() if p.requires_grad)
			# Add parameters for the shared soft prompt
			trainable_params += d * args1.n_virtual_tokens
			print(f"Number of trainable parameters in {enc1_name}: {trainable_params / 1e6:.2f}M")
			trainable_params = sum(p.numel() for p in enc2.parameters() if p.requires_grad)
			print(f"Number of trainable parameters in {enc2_name}: {trainable_params / 1e6:.2f}M")
		 
			
			# _args = PyTorchBenchmarkArguments(
			# 	load_my_custom_model=True,
			# 	models=[enc1_name, enc2_name],
			# 	custom_models={
			# 		enc1_name: enc1,
			# 		enc2_name: enc2,
			# 	},
			# 	batch_sizes=[1],
			# 	sequence_lengths=[325],
			# )
			#
			# print("Models being benchmarked: ", _args.models)
			#
			# benchmark = PyTorchBenchmark(_args, configs=[enc1.config, enc2.config])
			# results = benchmark.run()
			# print(results)
