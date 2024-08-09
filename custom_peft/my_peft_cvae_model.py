# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from . import __version__
from .config import PeftConfig
from .tuners import (
	AdaLoraModel,
	AdaptionPromptModel,
	IA3Model,
	LoraModel,
	PrefixEncoder,
	PromptEmbedding,
	PromptEncoder,
)
from .utils import (
	SAFETENSORS_WEIGHTS_NAME,
	TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
	WEIGHTS_NAME,
	PeftType,
	TaskType,
	_get_batch_size,
	_prepare_prompt_learning_config,
	_set_adapter,
	_set_trainable,
	add_library_to_model_card,
	get_peft_lib_model_state_dict,
	id_tensor_storage,
	infer_device,
	load_peft_weights,
	set_peft_lib_model_state_dict,
)

PEFT_TYPE_TO_MODEL_MAPPING = {
	PeftType.LORA: LoraModel,
	PeftType.PROMPT_TUNING: PromptEmbedding,
	PeftType.P_TUNING: PromptEncoder,
	PeftType.PREFIX_TUNING: PrefixEncoder,
	PeftType.ADALORA: AdaLoraModel,
	PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
	PeftType.IA3: IA3Model,
}


class PeftCvaeModel(PushToHubMixin, torch.nn.Module):
	"""
	Base model encompassing various Peft methods.

	Args:
		model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
		peft_config ([`PeftConfig`]): The configuration of the Peft model.
		adapter_name (`str`): The name of the adapter, defaults to `"default"`.

	**Attributes**:
		- **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer model used for Peft.
		- **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
		- **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
		saving the model.
		- **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
		using [`PromptLearningConfig`].
		- **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
		using [`PromptLearningConfig`].
		- **transformer_backbone_name** (`str`) -- The name of the transformer
		backbone in the base model if using [`PromptLearningConfig`].
		- **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
		in the base model if using [`PromptLearningConfig`].
	"""
	
	def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default"):
		super().__init__()
		self.modules_to_save = None
		self.active_adapter = adapter_name
		self.peft_type = peft_config.peft_type
		
		self._is_prompt_learning = peft_config.is_prompt_learning
		if self._is_prompt_learning:
			self._peft_config = {adapter_name: peft_config}
			self.base_model = model
			self.add_adapter(adapter_name, peft_config)
		else:
			self._peft_config = None
			cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
			self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
			self.set_additional_trainable_modules(peft_config, adapter_name)
		
		if getattr(model, "is_gradient_checkpointing", True):
			model = self._prepare_model_for_gradient_checkpointing(model)
		
		# the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
		# numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
		# behavior we disable that in this line.
		if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
			self.base_model.config.pretraining_tp = 1
	
	@property
	def peft_config(self) -> dict[str, PeftConfig]:
		if self._is_prompt_learning:
			return self._peft_config
		return self.base_model.peft_config
	
	@property
	def active_adapters(self) -> list[str]:
		try:
			adapters = self.base_model.active_adapters
		except AttributeError:
			adapters = self.active_adapter
			if isinstance(adapters, str):
				adapters = [adapters]
		return adapters
	
	@peft_config.setter
	def peft_config(self, value: dict[str, PeftConfig]):
		if self._is_prompt_learning:
			self._peft_config = value
		else:
			self.base_model.peft_config = value
	
	
	def save_pretrained(
			self,
			save_directory: str,
			safe_serialization: bool = False,
			selected_adapters: Optional[List[str]] = None,
			is_main_process: bool = True,
			**kwargs: Any,
	):
		r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
		if os.path.isfile(save_directory):
			raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
		
		if selected_adapters is None:
			selected_adapters = list(self.peft_config.keys())
		else:
			if any(
					selected_adapter_name not in list(self.peft_config.keys())
					for selected_adapter_name in selected_adapters
			):
				raise ValueError(
					f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
					f" {list(self.peft_config.keys())} - got {selected_adapters}."
				)
		
		if is_main_process:
			os.makedirs(save_directory, exist_ok=True)
			self.create_or_update_model_card(save_directory)
		
		for adapter_name in selected_adapters:
			peft_config = self.peft_config[adapter_name]
			# save only the trainable weights
			output_state_dict = get_peft_lib_model_state_dict(
				self,
				state_dict=kwargs.get("state_dict", None),
				adapter_name=adapter_name
			)
			output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
			os.makedirs(output_dir, exist_ok=True)
			
			if is_main_process and safe_serialization:
				# Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
				# Safetensors does not allow tensor aliasing.
				# We're going to remove aliases before saving
				ptrs = collections.defaultdict(list)
				for name, tensor in output_state_dict.items():
					# Sometimes in the state_dict we have non-tensor objects.
					# e.g. in bitsandbytes we have some `str` objects in the state_dict
					if isinstance(tensor, torch.Tensor):
						ptrs[id_tensor_storage(tensor)].append(name)
					else:
						# In the non-tensor case, fall back to the pointer of the object itself
						ptrs[id(tensor)].append(name)
				
				# These are all the pointers of shared tensors.
				shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
				
				for _, names in shared_ptrs.items():
					# Here we just clone the shared tensors to avoid tensor aliasing which is
					# not supported in safetensors.
					for shared_tensor_name in names[1:]:
						output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
				
				safe_save_file(
					output_state_dict,
					os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
					metadata={"format": "pt"},
				)
			elif is_main_process:
				torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
			
			# save the config and change the inference mode to `True`
			if peft_config.base_model_name_or_path is None:
				peft_config.base_model_name_or_path = (
					self.base_model.__dict__.get("name_or_path", None)
					if peft_config.is_prompt_learning
					else self.base_model.model.__dict__.get("name_or_path", None)
				)
			inference_mode = peft_config.inference_mode
			peft_config.inference_mode = True
			
			if peft_config.task_type is None:
				# deal with auto mapping
				base_model_class = self._get_base_model_class(
					is_prompt_tuning=peft_config.is_prompt_learning,
				)
				parent_library = base_model_class.__module__
				
				auto_mapping_dict = {
					"base_model_class": base_model_class.__name__,
					"parent_library": parent_library,
				}
			else:
				auto_mapping_dict = None
			
			if is_main_process:
				peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
			peft_config.inference_mode = inference_mode
	
	@classmethod
	def from_pretrained(
			cls,
			model: PreTrainedModel,
			model_id: Union[str, os.PathLike],
			adapter_name: str = "default",
			is_trainable: bool = False,
			config: Optional[PeftConfig] = None,
			**kwargs: Any,
	):
		r"""
		Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

		Note that the passed `model` may be modified inplace.

		Args:
			model ([`~transformers.PreTrainedModel`]):
				The model to be adapted. The model should be initialized with the
				[`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
			model_id (`str` or `os.PathLike`):
				The name of the PEFT configuration to use. Can be either:
					- A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
					  Hub.
					- A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
					  method (`./my_peft_config_directory/`).
			adapter_name (`str`, *optional*, defaults to `"default"`):
				The name of the adapter to be loaded. This is useful for loading multiple adapters.
			is_trainable (`bool`, *optional*, defaults to `False`):
				Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
				inference
			config ([`~peft.PeftConfig`], *optional*):
				The configuration object to use instead of an automatically loaded configuation. This configuration
				object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
				loaded before calling `from_pretrained`.
			kwargs: (`optional`):
				Additional keyword arguments passed along to the specific PEFT configuration class.
		"""
		from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
		
		# load the config
		if config is None:
			config = PEFT_TYPE_TO_CONFIG_MAPPING[
				PeftConfig._get_peft_type(
					model_id,
					subfolder=kwargs.get("subfolder", None),
					revision=kwargs.get("revision", None),
					cache_dir=kwargs.get("cache_dir", None),
					use_auth_token=kwargs.get("use_auth_token", None),
				)
			].from_pretrained(model_id, **kwargs)
		elif isinstance(config, PeftConfig):
			config.inference_mode = not is_trainable
		else:
			raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")
		
		if (getattr(model, "hf_device_map", None) is not None) and len(
				set(model.hf_device_map.values()).intersection({"cpu", "disk"})
		) > 0:
			remove_hook_from_submodules(model)
		
		if config.is_prompt_learning and is_trainable:
			raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
		else:
			config.inference_mode = not is_trainable
		
		if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
			model = cls(model, config, adapter_name)
		else:
			model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
		model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
		return model
	
	def _setup_prompt_encoder(self, adapter_name: str):
		config = self.peft_config[adapter_name]
		if not hasattr(self, "prompt_encoder"):
			self.prompt_encoder = torch.nn.ModuleDict({})
			self.prompt_tokens = {}
		transformer_backbone = None
		for name, module in self.base_model.named_children():
			for param in module.parameters():
				param.requires_grad = False
			if isinstance(module, PreTrainedModel):
				# Make sure to freeze Transformers model
				if transformer_backbone is None:
					transformer_backbone = module
					self.transformer_backbone_name = name
		if transformer_backbone is None:
			transformer_backbone = self.base_model
		
		if config.num_transformer_submodules is None:
			config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
		
		for named_param, value in list(transformer_backbone.named_parameters()):
			# for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape [0]
			# the actual unsharded shape is stored in "ds_shape" attribute
			# special handling is needed in case the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig
			# has been called before
			# For reference refer to issue: https://github.com/huggingface/peft/issues/996
			deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)
			
			if value.shape[0] == self.base_model.config.vocab_size or (
					deepspeed_distributed_tensor_shape is not None
					and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
			):
				self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
				break
		
		if config.peft_type == PeftType.PROMPT_TUNING:
			prompt_encoder = PromptEmbedding(config, self.word_embeddings)
		elif config.peft_type == PeftType.P_TUNING:
			prompt_encoder = PromptEncoder(config)
		elif config.peft_type == PeftType.PREFIX_TUNING:
			prompt_encoder = PrefixEncoder(config)
		else:
			raise ValueError("Not supported")
		
		prompt_encoder = prompt_encoder.to(self.device)
		self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
		self.prompt_tokens[adapter_name] = torch.arange(
			config.num_virtual_tokens * config.num_transformer_submodules
		).long()
	
	def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
		r"""
		Prepares the model for gradient checkpointing if necessary
		"""
		if not (
				getattr(model, "is_loaded_in_8bit", False)
				or getattr(model, "is_loaded_in_4bit", False)
				or getattr(model, "is_quantized", False)
		):
			if hasattr(model, "enable_input_require_grads"):
				model.enable_input_require_grads()
			elif hasattr(model, "get_input_embeddings"):
				
				def make_inputs_require_grad(module, input, output):
					output.requires_grad_(True)
				
				model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
		return model
	
	def get_prompt_embedding_to_save(self, adapter_name: str):
		"""
		Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
		PeftType.LORA`.
		"""
		# ##################################################################################################### #
		# ################################################ My custom  ######################################### #
		# ##################################################################################################### #
		config = self.peft_config[adapter_name]
		prompt_encoder = self.prompt_encoder['{}'.format(adapter_name)]
		prompt_tokens = (
			self.prompt_tokens['{}'.format(adapter_name)].unsqueeze(0).expand(1, -1).to(
				prompt_encoder.embedding.weight.device)
		)
		if config.peft_type == PeftType.PREFIX_TUNING:
			prompt_tokens = prompt_tokens[:, : config.num_virtual_tokens]
		prompt_embeddings = prompt_encoder(prompt_tokens)
		prompt_embeddings = prompt_embeddings[0].detach().cpu()
		
		return prompt_embeddings
	
	def get_prompt(self, batch_size: int):
		"""
		Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
		Added functionality: Returns the virtual prompts for specific library
		"""
		peft_config = self.active_peft_config
		
		# ##################################################################################################### #
		# ################################################ My custom  ######################################### #
		# ##################################################################################################### #
		
		prompt_encoder = self.prompt_encoder[f'{self.active_adapter}']
		prompt_tokens = (
			self.prompt_tokens[f'{self.active_adapter}']
			.unsqueeze(0)
			.expand(batch_size, -1)
			.to(prompt_encoder.embedding.weight.device)
		)
		if peft_config.peft_type == PeftType.PREFIX_TUNING:
			prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
			if peft_config.inference_mode:
				past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
			else:
				past_key_values = prompt_encoder(prompt_tokens)
			if self.base_model_torch_dtype is not None:
				past_key_values = past_key_values.to(self.base_model_torch_dtype)
			past_key_values = past_key_values.view(
				batch_size,
				peft_config.num_virtual_tokens,
				peft_config.num_layers * 2,
				peft_config.num_attention_heads,
				peft_config.token_dim // peft_config.num_attention_heads,
			)
			if peft_config.num_transformer_submodules == 2:
				past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
			past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
				peft_config.num_transformer_submodules * 2
			)
			if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
				post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
				past_key_values = post_process_fn(past_key_values)
			return past_key_values
		else:
			if peft_config.inference_mode:
				prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
			else:
				prompts = prompt_encoder(prompt_tokens)
			return prompts
	
	def get_nb_trainable_parameters(self):
		r"""
		Returns the number of trainable parameters and number of all parameters in the model.
		"""
		trainable_params = 0
		all_param = 0
		for _, param in self.named_parameters():
			num_params = param.numel()
			# if using DS Zero 3 and the weights are initialized empty
			if num_params == 0 and hasattr(param, "ds_numel"):
				num_params = param.ds_numel
			
			# Due to the design of 4bit linear layers from bitsandbytes
			# one needs to multiply the number of parameters by 2 to get
			# the correct number of parameters
			if param.__class__.__name__ == "Params4bit":
				num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
				num_params = num_params * 2 * num_bytes
			
			all_param += num_params
			if param.requires_grad:
				trainable_params += num_params
		
		return trainable_params, all_param
	
	def print_trainable_parameters(self):
		"""
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
		trainable_params, all_param = self.get_nb_trainable_parameters()
		
		print(
			f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
		)
	
	def __getattr__(self, name: str):
		"""Forward missing attributes to the wrapped module."""
		try:
			return super().__getattr__(name)  # defer to nn.Module's logic
		except AttributeError:
			return getattr(self.base_model, name)
	
	def forward(self, *args: Any, **kwargs: Any):
		"""
		Forward pass of the model.
		"""
		return self.get_base_model()(*args, **kwargs)
	
	def _get_base_model_class(self, is_prompt_tuning=False):
		"""
		Returns the base model class.
		"""
		if not is_prompt_tuning:
			return self.base_model.model.__class__
		return self.base_model.__class__
	
	@contextmanager
	def disable_adapter(self):
		"""
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
		try:
			if self.peft_config[self.active_adapter].is_prompt_learning:
				# TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
				# letting the underyling methods deal with it, same as how LoRA does it.
				old_forward = self.forward
				self.forward = self.base_model.forward
				old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
				self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
			else:
				self.base_model.disable_adapter_layers()
			yield
		finally:
			if self.peft_config[self.active_adapter].is_prompt_learning:
				self.forward = old_forward
				self.old_prepare_inputs_for_generation = old_prepare_inputs_for_generation
			else:
				self.base_model.enable_adapter_layers()
	
	def get_base_model(self):
		"""
		Returns the base model.
		"""
		return self.base_model if self.active_peft_config.is_prompt_learning else self.base_model.model
	
	def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
		"""
		        Add an adapter to the model based on the passed configuration.

		        The name for the new adapter should be unique.

		        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
		        adapter.

		        Args:
		            adapter_name (`str`):
		                The name of the adapter to be added.
		            peft_config ([`PeftConfig`]):
		                The configuration of the adapter to be added.
		"""
		if peft_config.peft_type != self.peft_type:
			raise ValueError(
				f"Cannot combine adapters with different peft types. "
				f"Found {self.peft_type} and {peft_config.peft_type}."
			)
		
		self.peft_config[adapter_name] = peft_config  # We will use same peft config for all libraries
		
		try:
			if peft_config.is_prompt_learning:
				if hasattr(self.config, "to_dict"):
					dict_config = self.config.to_dict()
				else:
					dict_config = self.config
				
				peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
				self._setup_prompt_encoder(adapter_name)
			elif peft_config.is_adaption_prompt:
				self.base_model.add_adapter(adapter_name, peft_config)
			else:
				self.base_model.inject_adapter(self, adapter_name)
		
		except Exception:  # somthing went wrong, roll back
			del self.peft_config[adapter_name]
			raise
		
		self.set_additional_trainable_modules(peft_config, adapter_name)
	
	def set_additional_trainable_modules(self, peft_config, adapter_name):
		if getattr(peft_config, "modules_to_save", None) is not None:
			if self.modules_to_save is None:
				self.modules_to_save = set(peft_config.modules_to_save)
			else:
				self.modules_to_save.update(peft_config.modules_to_save)
			_set_trainable(self, adapter_name)
	
	@classmethod
	def _split_kwargs(cls, kwargs: Dict[str, Any]):
		_kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
		hf_hub_download_kwargs = {}
		other_kwargs = {}
		
		for key, value in kwargs.items():
			if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
				hf_hub_download_kwargs[key] = value
			else:
				other_kwargs[key] = value
		
		return hf_hub_download_kwargs, other_kwargs
	
	def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any):
		"""
		        Load a trained adapter into the model.

		        The name for the new adapter should be unique.

		        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
		        adapter.

		        Args:
		            adapter_name (`str`):
		                The name of the adapter to be added.
		            peft_config ([`PeftConfig`]):
		                The configuration of the adapter to be added.
		            is_trainable (`bool`, *optional*, defaults to `False`):
		                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
		                used for inference.
		            kwargs: (`optional`):
		                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
		        """
		from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING
		
		hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
		torch_device = infer_device()
		
		if adapter_name not in self.peft_config:
			# load the config
			peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
				PeftConfig._get_peft_type(
					model_id,
					**hf_hub_download_kwargs,
				)
			].from_pretrained(
				model_id,
				**hf_hub_download_kwargs,
			)
			if peft_config.is_prompt_learning and is_trainable:
				raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
			else:
				peft_config.inference_mode = not is_trainable
			self.add_adapter(adapter_name, peft_config)
		
		adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)
		
		# load the weights into the model
		load_result = set_peft_lib_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
		if (
				(getattr(self, "hf_device_map", None) is not None)
				and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
				and len(self.peft_config) == 1
		):
			device_map = kwargs.get("device_map", "auto")
			max_memory = kwargs.get("max_memory", None)
			offload_dir = kwargs.get("offload_folder", None)
			offload_index = kwargs.get("offload_index", None)
			
			dispatch_model_kwargs = {}
			# Safety checker for previous `accelerate` versions
			# `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
			if "offload_index" in inspect.signature(dispatch_model).parameters:
				dispatch_model_kwargs["offload_index"] = offload_index
			
			no_split_module_classes = self._no_split_modules
			
			if device_map != "sequential":
				max_memory = get_balanced_memory(
					self,
					max_memory=max_memory,
					no_split_module_classes=no_split_module_classes,
					low_zero=(device_map == "balanced_low_0"),
				)
			if isinstance(device_map, str):
				device_map = infer_auto_device_map(
					self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
				)
			dispatch_model(
				self,
				device_map=device_map,
				offload_dir=offload_dir,
				**dispatch_model_kwargs,
			)
			hook = AlignDevicesHook(io_same_device=True)
			if self.peft_config[adapter_name].is_prompt_learning:
				remove_hook_from_submodules(self.prompt_encoder)
			add_hook_to_module(self.get_base_model(), hook)
		
		# Set model in evaluation mode to deactivate Dropout modules by default
		if not is_trainable:
			self.eval()
		return load_result
	
	def set_adapter(self, adapter_name: str):
		"""
        Sets the active adapter.

        Only one adapter can be active at a time.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
		if adapter_name not in self.peft_config:
			raise ValueError(f"Adapter {adapter_name} not found.")
		self.active_adapter = adapter_name
		if not self.peft_config[adapter_name].is_prompt_learning:
			self.base_model.set_adapter(adapter_name)
		_set_adapter(self, adapter_name)
	
	@property
	def base_model_torch_dtype(self):
		return getattr(self.base_model, "dtype", None)
	
	@property
	def active_peft_config(self):
		return self.peft_config[self.active_adapter]
	
	def create_or_update_model_card(self, output_dir: str):
		"""
		Updates or create model card to include information about peft:
		1. Adds `peft` library tag
		2. Adds peft version
		3. Adds base model info
		4. Adds quantization information if it was used
		"""
		
		filename = os.path.join(output_dir, "README.md")
		
		card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())
		
		card.data["library_name"] = "peft"
		
		model_config = getattr(self, "config", None)
		if hasattr(model_config, "to_dict"):
			model_config = model_config.to_dict()
		if model_config is not None and "_name_or_path" in model_config:
			card.data["base_model"] = model_config["_name_or_path"]
		
		lines = card.text.splitlines()
		
		quantization_config = None
		if hasattr(model_config, "quantization_config"):
			quantization_config = self.config.quantization_config.to_dict()
		training_config_text = ""
		quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
		# Adds quantization information if it was used
		if quantization_config is not None:
			training_config_text += f"\n{quantization_prefix}\n"
			training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
			training_config_text += "\n"
		
		training_procedure_heading = "## Training procedure"
		if quantization_prefix not in lines and bool(training_config_text):
			if training_procedure_heading in lines:
				lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
			else:
				lines.append(f"{training_procedure_heading}\n{training_config_text}")
		
		# Adds peft version
		framework_block_heading = "### Framework versions"
		if f"- PEFT {__version__}" not in lines:
			if framework_block_heading in lines:
				lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}")
			else:
				lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}")
		
		card.text = "\n".join(lines)
		card.save(filename)


class PeftCvaeModelForCausalLM(PeftCvaeModel):
	"""
	Peft model for causal language modeling.

	Args:
		model ([`~transformers.PreTrainedModel`]): Base transformer model.
		peft_config ([`PeftConfig`]): Peft config.

	"""
	
	def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
		super().__init__(model, peft_config, adapter_name)
		# # Store the base model's prepare_inputs_for_generation method (use to revert when there is an error)
		self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
		
		# Initialize the latent_attention_weights <- [Hack] will be used during generate API, not in FWD pass
		self.latent_attention_weights = None
	
	def forward(
			self,
			latent_attention_weights=None,
			input_ids=None,
			attention_mask=None,
			inputs_embeds=None,
			labels=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
			task_ids=None,
			**kwargs,
	):
		
		peft_config = self.active_peft_config
		if not peft_config.is_prompt_learning:
			if self.base_model.config.model_type == "mpt":
				if inputs_embeds is not None:
					raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
				return self.base_model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					labels=labels,
					output_attentions=output_attentions,
					output_hidden_states=output_hidden_states,
					return_dict=return_dict,
					**kwargs,
				)
			
			return self.base_model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				inputs_embeds=inputs_embeds,
				labels=labels,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				**kwargs,
			)
		
		batch_size = _get_batch_size(input_ids, inputs_embeds)
		if attention_mask is not None:
			
			# # For prefix
			prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
			attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
		
		if kwargs.get("position_ids", None) is not None:
			warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
			kwargs["position_ids"] = None
		if kwargs.get("token_type_ids", None) is not None:
			warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
			kwargs["token_type_ids"] = None
		kwargs.update(
			{
				"attention_mask": attention_mask,
				"labels": labels,
				"output_attentions": output_attentions,
				"output_hidden_states": output_hidden_states,
				"return_dict": return_dict,
			}
		)
		
		if peft_config.peft_type == PeftType.PREFIX_TUNING:
			past_key_values = self.get_prompt(batch_size)
			return self.base_model(
				input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
			)
		else:
			if inputs_embeds is None:
				inputs_embeds = self.word_embeddings(input_ids)
			# concat prompt labels
			if labels is not None:
				prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
				kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
			prompts = self.get_prompt(batch_size=batch_size)  # Added functionality
			prompts = prompts.to(inputs_embeds.dtype)
			
			# [My Custom] Weigh the prompts with the latent attention weights
			if latent_attention_weights is None:
				raise ValueError("latent_attention_weights is required for cVAE-based prefix tuning")
			
			# For shared prompts across instances
			prompts = prompts * latent_attention_weights
			
			# For no sharing
			# prompts = latent_attention_weights
			
			# # For Prefix
			inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
			
			return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
	
	def generate(self, **kwargs):
		# # This will override base transformer's prepare_inputs_for_generation() method
		# # and PEFT's prepare_inputs_for_generation() will be called instead but we have saved the base model's
		self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
		
		if hasattr(self.base_model, "model"):
			self.base_model.model.generation_config = self.generation_config
		else:
			self.base_model.generation_config = self.generation_config
		
		try:
			outputs = self.base_model.generate(**kwargs)
		except:
			self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
			raise
		else:
			self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
			return outputs
	
	# ##################################################################################################### #
	# ################################################ My custom  ######################################### #
	# ##################################################################################################### #
	def prepare_inputs_for_generation(self, *args, **kwargs):
		peft_config = self.active_peft_config
		
		# # First, get the base model's inputs -> Its prepare inputs for generation method is stored here
		model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
		
		# https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
		# for some architectures which requires a special fix for prompt tuning etc.
		# TODO: starting with transformers 4.38, all architectures should support caching.
		uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
		uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
		transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
		uses_cache = uses_transformers_4_38 or (
				uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
		)
		
		if peft_config.is_prompt_learning:
			
			if uses_cache and (model_kwargs["past_key_values"] is not None):
				# change in the logic of `prepare_inputs_for_generation` makes the below code necessary
				# In prompt learning methods, past key values are longer when compared to the `input_ids`.
				# As such only consider the last input ids in the autogressive generation phase.
				if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
					model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]
			
			if model_kwargs.get("attention_mask", None) is not None:
				size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
				
				# # For Prefix
				prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
				model_kwargs["attention_mask"] = torch.cat(
					(prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
				)
				
			
			if model_kwargs.get("position_ids", None) is not None:
				warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
				model_kwargs["position_ids"] = None
			
			if kwargs.get("token_type_ids", None) is not None:
				warnings.warn(
					"Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
				)
				kwargs["token_type_ids"] = None
			
			if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
				past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
				
				# Added functionality
				if self.latent_attention_weights is not None:
					past_key_values = past_key_values * self.latent_attention_weights
				
				model_kwargs["past_key_values"] = past_key_values
			
			else:
				if model_kwargs["past_key_values"] is None:
					batch_size = model_kwargs["input_ids"].shape[0]
					inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
					prompts = self.get_prompt(batch_size=batch_size)
					prompts = prompts.to(inputs_embeds.dtype)

					# Added functionality
					# For cases like beam-decoding, soft prompt encoded for a sample must be repeated for all beams
					# latent_attention_weights = self.latent_attention_weights.repeat(batch_size, 1, 1)
					# prompts = prompts * latent_attention_weights
					
					# For debugging
					prompts = prompts * self.latent_attention_weights
	
					# # For Prefix
					model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
					
					model_kwargs["input_ids"] = None
		
		# For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
		# passed in the forward pass to keep track of the position ids of the cache. We have to
		# pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
		# `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
		_ = model_kwargs.pop("cache_position", None)
		
		return model_kwargs
