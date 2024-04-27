import argparse
import copy
from custom_benchmark.benchmark import PyTorchBenchmark
from custom_benchmark.benchmark_args import PyTorchBenchmarkArguments
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator


# _args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[1], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(_args)
# results = benchmark.run()
# print(results)

n_virtual_tokens_exp = [50]
for t in n_virtual_tokens_exp:
	print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
	print(f"==================== Benchmarking for n_virtual_tokens = {t} ====================")
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
	parser = argparse.ArgumentParser()
	parser.add_argument("--enc_model_type", type=str)
	parser.add_argument('--n_virtual_tokens', type=int, default=t)
	parser.add_argument('--word_embedding_dim', type=int, default=1024)
	args = parser.parse_args()
	
	args1 = copy.deepcopy(args)
	args1.enc_model_type = "codebert-base"
	enc1 = LatentPromptAttentionGenerator(args1, args1.n_virtual_tokens, args1.word_embedding_dim)
	
	args2 = copy.deepcopy(args)
	args2.enc_model_type = 'codegen-350M'
	enc2 = IDPGSoftPromptGenerator(args2, args2.n_virtual_tokens, args2.word_embedding_dim)
	enc1_name = str(enc1)
	enc2_name = str(enc2)
	
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
