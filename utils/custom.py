import gc
import logging
import multiprocessing
import pathlib
import random
from typing import List

import GPUtil
import numpy as np
import torch
from pynvml import *
from transformers import set_seed

logger = logging.getLogger(__name__)


def unwrap_model(model):
	"""Unwrap the model from the DataParallel wrapper."""
	if isinstance(model, torch.nn.DataParallel):
		return model.module
	return model


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


def check_available_gpu_memory():
	# @title Check available memory of GPU
	# Check that we are using 100% of GPU
	# memory footprint support libraries/code
	# Required Libraries: psutil, humanize, gputil, py3nvml (to track GPU memory usage)
	import psutil
	import humanize
	import os
	import GPUtil as GPU
	GPUs = GPU.getGPUs()
	# XXX: only one GPU on Colab and isn’t guaranteed
	gpu = GPUs[0]
	process = psutil.Process(os.getpid())
	print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
		  " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
	print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
																								gpu.memoryUsed,
																								gpu.memoryUtil * 100,
																								gpu.memoryTotal))


def is_rank_0() -> bool:
	# Can also use accelerator.is_local_main_process if using Accelerator
	return int(os.environ.get("RANK", "0")) == 0


def log_dist(
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


def set_dist(args):
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
	set_seed(args.seed)


