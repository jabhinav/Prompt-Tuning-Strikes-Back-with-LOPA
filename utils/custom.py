import gc
import json
import logging
import multiprocessing
import os
import pathlib
import GPUtil
import random
from typing import List, Dict
from pynvml import *

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def b2mb(x):
	"""Convert bytes to megabytes."""
	return int(x / 2 ** 20)


class TorchTracemalloc:
	"""
	This context manager is used to track the peak memory usage of the process
	"""
	
	def __enter__(self):
		gc.collect()
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()  # reset the peak gauge to zero [imp to compare relative memory usage]
		self.begin = torch.cuda.memory_allocated()
		return self
	
	def __exit__(self, *exc):
		gc.collect()
		torch.cuda.empty_cache()
		self.end = torch.cuda.memory_allocated()
		self.peak = torch.cuda.max_memory_allocated()
		self.used = b2mb(self.end - self.begin)
		self.peaked = b2mb(self.peak - self.begin)
		# print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def debug_memory(msg: str = "", device=0, accelerator=None):
	"""Print memory usage."""
	if accelerator is not None:
		
		# To report memory consumption once per machine
		if accelerator.is_local_main_process:
			GPUtil.showUtilization()
			try:
				nvmlInit()  # this library allows to access the nvidia-smi information in Python directly
				handle = nvmlDeviceGetHandleByIndex(device)
				info = nvmlDeviceGetMemoryInfo(handle)
				logger.info(f"[MEMORY USAGE {device}] {msg}: {info.used // 1024 ** 2} MB.")
			except NVMLError as error:
				logger.info(f'[MEMORY USAGE {device}] {msg}: {torch.cuda.memory_allocated() / 1024 ** 3} GB')
	else:
		GPUtil.showUtilization()
		try:
			nvmlInit()  # this library allows to access the nvidia-smi information in Python directly
			handle = nvmlDeviceGetHandleByIndex(device)
			info = nvmlDeviceGetMemoryInfo(handle)
			logger.info(f"[MEMORY USAGE {device}] {msg}: {info.used // 1024 ** 2} MB.")
		except NVMLError as error:
			logger.info(f'[MEMORY USAGE {device}] {msg}: {torch.cuda.memory_allocated() / 1024 ** 3} GB')


def is_rank_0() -> bool:
	return int(os.environ.get("RANK", "0")) == 0


def log_dist(
		logger,
		message: str,
		ranks: List[int],
		level: int = logging.INFO
) -> None:
	"""Log messages for specified ranks only"""
	my_rank = int(os.environ.get("RANK", "0"))
	if my_rank in ranks:
		if level == logging.INFO:
			logger.info(f'[Rank {my_rank}] {message}')
		if level == logging.ERROR:
			logger.error(f'[Rank {my_rank}] {message}')
		if level == logging.DEBUG:
			logger.debug(f'[Rank {my_rank}] {message}')


def create_dir(path: str):
	if not is_rank_0():
		return
	_dir = pathlib.Path(path)
	_dir.mkdir(parents=True, exist_ok=True)


def set_dist(args, logger):
	
	# To train on cpu, set args.no_cuda=True else it will use all available gpus [Recommended use for now]
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	
	# To enable distributed training (does it mean multi-node?), set local_rank
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		# noinspection PyUnresolvedReferences
		torch.distributed.init_process_group(backend='nccl')
		args.local_rank += args.node_index * args.gpu_per_node
		args.n_gpu = 1
	
	cpu_cont = multiprocessing.cpu_count()  # Gives number of logical CPU cores
	# Do not use all cpu cores for parallel processing. For computationally intensive tasks, recommended usage is
	# to use number of physical CPU cores i.e. = (number of logical CPU cores)/2
	# Recommended reading: https://superfastpython.com/multiprocessing-pool-num-workers/
	args.cpu_cont = cpu_cont - int(cpu_cont / 2)  # Ignore half of the cores
	args.device = device
	
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count(using): %d, "
				   "cpu count(available): %d", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1),
				   args.cpu_cont, cpu_cont)


def set_seed(args):
	"""set random seed."""
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def save_predictions_mbxp_format(args, logger, output: Dict[str, Dict[str, List[str]]], lang='python', d_type='MBPP'):
	"""
	Save the predictions in the format required by the MBXP evaluation script.
	:param args:
	:param logger:
	:param output:
	:param lang:
	:param d_type:
	:return:
	"""
	
	# Save each library's predictions in a separate file
	for k in range(args.num_libraries):
		with open(os.path.join(args.log_dir, f'mbxp_solutions_lib_{k}.json'), 'w') as file:
			for problem in output:
				for response in output[problem][f'lib_{k}']:
					result_dict: dict = {
						"task_id":  problem,
						"language": lang,
						"completion": response,
					}
					file.write(json.dumps(result_dict) + '\n')
					
		logger.info(f"Saved predictions for library {k} in the format required by the MBXP evaluation script")
		
	# Save all the predictions in a single file
	with open(os.path.join(args.log_dir, f'mbxp_solutions.json'), 'w') as file:
		for problem in output:
			for k in range(args.num_libraries):
				for response in output[problem][f'lib_{k}']:
					result_dict: dict = {
						"task_id":  problem,
						"language": lang,
						"completion": response,
					}
					file.write(json.dumps(result_dict) + '\n')
					
	logger.info(f"Saved all predictions in a single file in the format required by the MBXP evaluation script")
