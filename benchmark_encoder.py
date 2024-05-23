import argparse
import copy
from custom_benchmark.benchmark import PyTorchBenchmark
from custom_benchmark.benchmark_args import PyTorchBenchmarkArguments
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator


# _args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[1], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(_args)
# results = benchmark.run()
# print(results)

do_mem_runtime_benchmark = False


word_embedding_dims = [1024, 2048, 4096, 2560, 3072, 4096]  # For each foundation model presented in the paper
ranks = [4, 2, 1]
token_lens = [5, 10, 25, 50, 100]
for d in word_embedding_dims:
	for t in token_lens:
		for r in ranks:
			print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
			print(f"==================== Benchmarking for d/r/t = {d}/{r}/{t} ====================")
			print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
			parser = argparse.ArgumentParser()
			parser.add_argument('--lp_rank', type=int, default=r)
			parser.add_argument('--n_virtual_tokens', type=int, default=t)
			parser.add_argument('--word_embedding_dim', type=int, default=d)
			parser.add_argument("--enc_model_type", type=str, default="codesage-small",
								choices=["codebert-base", "codet5p-110m-embedding", "codesage-small", "codesage-base", "codesage-large"])
			args = parser.parse_args()
			
			args1 = copy.deepcopy(args)
			enc1 = LatentPromptAttentionGenerator(args1, args1.n_virtual_tokens, args1.word_embedding_dim)
			
			args2 = copy.deepcopy(args)
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
		 
			
			if do_mem_runtime_benchmark:
				_args = PyTorchBenchmarkArguments(
					load_my_custom_model=True,
					models=[enc1_name, enc2_name],
					custom_models={
						enc1_name: enc1,
						enc2_name: enc2,
					},
					batch_sizes=[1],
					sequence_lengths=[325],
				)
	
				print("Models being benchmarked: ", _args.models)
	
				benchmark = PyTorchBenchmark(_args, configs=[enc1.config, enc2.config])
				results = benchmark.run()
				print(results)
