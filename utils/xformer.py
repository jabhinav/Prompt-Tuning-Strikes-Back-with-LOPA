import logging
import torch
import numpy as np
from transformers import RobertaConfig, T5Config, BartConfig, GPT2Config, OpenAIGPTConfig, BertConfig, \
	DistilBertConfig, GPTNeoConfig, AutoConfig
from transformers import RobertaModel, T5ForConditionalGeneration, BartForConditionalGeneration, GPT2LMHeadModel, \
	OpenAIGPTLMHeadModel, BertForMaskedLM, DistilBertForMaskedLM, GPTNeoForCausalLM, AutoModel, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5Tokenizer, BartTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer, \
	BertTokenizer, DistilBertTokenizer, AutoTokenizer, CodeGenTokenizer, CodeGenTokenizerFast
from utils.custom import is_rank_0

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
	'roberta': (RobertaConfig, RobertaModel),
	't5': (T5Config, T5ForConditionalGeneration),
	'bart': (BartConfig, BartForConditionalGeneration),
	'gpt2': (GPT2Config, GPT2LMHeadModel),
	'gpt2-large': (GPT2Config, GPT2LMHeadModel),
	'gpt2-xl': (GPT2Config, GPT2LMHeadModel),
	'gpt-neo-125M': (GPTNeoConfig, GPTNeoForCausalLM),
	'gpt-neo-1.3B': (GPTNeoConfig, GPTNeoForCausalLM),
	'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel),
	'bert': (BertConfig, BertForMaskedLM),
	'distilbert': (DistilBertConfig, DistilBertForMaskedLM),
	'codebert-base': (AutoConfig, AutoModel),
	# ############################# Codesage encoder-only Models ############################## #
	'codesage-base': (AutoConfig, AutoModel),
	'codesage-small': (AutoConfig, AutoModel),
	'codesage-large': (AutoConfig, AutoModel),
	# ############################# Microsoft Phi Models ############################## #
	'phi-1': (AutoConfig, AutoModelForCausalLM),
	'phi-1_5': (AutoConfig, AutoModelForCausalLM),
	'phi-2': (AutoConfig, AutoModelForCausalLM),
	'phi-3': (AutoConfig, AutoModelForCausalLM),
	'phi-3-small': (AutoConfig, AutoModelForCausalLM),
	'phi-3-medium': (AutoConfig, AutoModelForCausalLM),
	# ############################# Salesforce CodeT5 Models ############################# #
	'codet5-small': (T5Config, T5ForConditionalGeneration),
	'codet5-base': (T5Config, T5ForConditionalGeneration),
	'codet5-large': (T5Config, T5ForConditionalGeneration),
	'codet5-large-ntp-py': (T5Config, T5ForConditionalGeneration),
	'codet5p-110m-embedding': (AutoConfig, AutoModel),
	'codet5p-220m': (AutoConfig, T5ForConditionalGeneration),
	'codet5p-770m': (AutoConfig, T5ForConditionalGeneration),
	'codet5p-2b': (AutoConfig, T5ForConditionalGeneration),
	'codet5p-6b':  (AutoConfig, T5ForConditionalGeneration),
	# ############################# Salesforce CodeGen Models ############################# #
	'codegen-350M': (AutoConfig, AutoModelForCausalLM),
	'codegen-2B': (AutoConfig, AutoModelForCausalLM),
	'codegen-6B': (AutoConfig, AutoModelForCausalLM),
	'codegen-16B': (AutoConfig, AutoModelForCausalLM),
	'codegen2-1B': (AutoConfig, AutoModelForCausalLM),
	'codegen2-3_7B': (AutoConfig, AutoModelForCausalLM),
	'codegen2-7B': (AutoConfig, AutoModelForCausalLM),
	'codegen2-16B': (AutoConfig, AutoModelForCausalLM),
	'codegen25-7b-multi': (AutoConfig, AutoModelForCausalLM),  # Latest CodeGen2.5 7B model (Aug 2023)
	# ############################# Meta LLama Models ############################# #
	'CodeLlama-7b-Python-hf': (AutoConfig, AutoModelForCausalLM),
	'CodeLlama-13b-Python-hf': (AutoConfig, AutoModelForCausalLM),
	'CodeLlama-34b-Python-hf': (AutoConfig, AutoModelForCausalLM),
	'Meta-Llama-3-8B': (AutoConfig, AutoModelForCausalLM),
	'Meta-Llama-3-70B': (AutoConfig, AutoModelForCausalLM),
	# ############################# DeepSeek-Coder Models ############################# #
	'deepseek-coder-1.3b-base': (AutoConfig, AutoModelForCausalLM),
	'deepseek-coder-7b-base': (AutoConfig, AutoModelForCausalLM),
	'deepseek-coder-33b-base': (AutoConfig, AutoModelForCausalLM),
}

TOKENIZER_CLASSES = {
	'roberta': RobertaTokenizer,
	't5': T5Tokenizer,
	'codet5-small': RobertaTokenizer,
	'codet5-base': RobertaTokenizer,
	'codet5-large': RobertaTokenizer,
	# Official Documentation uses AutoTokenizer, but it is the same as RobertaTokenizer.
	# We want the same tokenization for all our models.
	'codet5-large-ntp-py': RobertaTokenizer,  # Official Documentation uses AutoTokenizer
	'bart': BartTokenizer,
	'gpt2': GPT2Tokenizer,
	'gpt2-xl': GPT2Tokenizer,
	'gpt-neo-125M': GPT2Tokenizer,
	'gpt-neo-1.3B': GPT2Tokenizer,
	'openai-gpt': OpenAIGPTTokenizer,
	'bert': BertTokenizer,
	'distilbert': DistilBertTokenizer,
}

LORA_IA3_TARGET_MODULES = {
	# ############################# Microsoft Phi Models ############################## #
	"phi-2": {
		"target_modules_lora": ["q_proj", "k_proj", "v_proj"],
	},
	"phi-3": {
		"target_modules_lora": ["qkv_proj"],
	},
	# ############################# Salesforce CodeT5 Models ############################# #
    "codet5p-220m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-770m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-2b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codet5p-6b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
	# ############################# Salesforce CodeGen Models ############################# #
	"codegen-350M": {
		"target_modules_lora": ["qkv_proj"],
		"target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
		"ff_modules": ["fc_in", "fc_out"]
	},
    "codegen2-1B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-3_7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
	# ############################# DeepSeek-Coder Models ############################# #
	"deepseek-coder-1.3b-base": {
		"target_modules_lora": ["q_proj", "k_proj", "v_proj"],
	},
	"deepseek-coder-7b-base": {
		"target_modules_lora": ["q_proj", "k_proj", "v_proj"],
	},
	# ############################# Meta LLama Models ############################# #
    "CodeLlama-7b-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Instruct-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-13b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-34b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
	"Meta-Llama-3-8B": {
		"target_modules_lora": ["q_proj", "k_proj", "v_proj"],
	},
}


def get_model_size(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	model_size = sum([np.prod(p.size()) for p in model_parameters])
	return "{}M".format(round(model_size / 1e+6))


def load_tokenizer(model_type, tokenizer_name):
	
	if model_type in TOKENIZER_CLASSES:
		tokenizer_class = TOKENIZER_CLASSES[model_type]
	else:
		tokenizer_class = AutoTokenizer
		if is_rank_0():
			print("Using AutoTokenizer for model_type: ", model_type)
	
	tokenizer = tokenizer_class.from_pretrained(
		tokenizer_name,
		trust_remote_code=True
	)
	
	if not tokenizer.eos_token:
		if tokenizer.bos_token:
			tokenizer.eos_token = tokenizer.bos_token
			tokenizer.eos_token_id = tokenizer.bos_token_id
			logger.info("bos_token used as eos_token")
		else:
			raise ValueError("No eos_token or bos_token found")
	
	# Some Tokenizers do not have pad_token. We add it here. (It will only be used for ease of use in my pipeline.)
	if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
		tokenizer.pad_token = tokenizer.eos_token
	
	if is_rank_0():
		logger.info("Finish loading Tokenizer from %s", tokenizer_name)
	return tokenizer


def load_base_model(model_type, config_name, model_path, load_in_8bit: bool = False):
	config_class, model_class = MODEL_CLASSES[model_type]
	
	config = config_class.from_pretrained(
		config_name if config_name else model_path,
		trust_remote_code=True,
		revision="main"
	)
	model = model_class.from_pretrained(
		model_path,
		trust_remote_code=True,
		revision="main",
		device_map="auto",
		# # For loading model in bfloat16, set. This is not quantization so it will not be as slow.
		# torch_dtype=torch.bfloat16,
		# # For loading model in 8bit, set. This is quantization so it will be slower.
		# load_in_8bit=True,
	)
	
	if is_rank_0():
		logger.info("Finish loading Base model [%s] from %s", get_model_size(model), model_path)
	return config, model


def get_huggingface_path(model: str) -> str:
	# ############################# OpenAI GPT Models ############################# #
	if model == 'gpt2':  # gpt2 (124M)
		huggingface_path = 'gpt2'
	elif model == 'gpt2-large':  # gpt2-medium(335M), gpt2-large (774M)
		huggingface_path = 'gpt2-large'
	elif model == 'gpt2-xl':
		huggingface_path = 'gpt2-xl'  # gpt2-xl (1.5B)
	elif model == 'gpt-neo-125M':
		huggingface_path = 'EleutherAI/gpt-neo-125M'
	elif model == 'gpt-neo-1.3B':
		huggingface_path = 'EleutherAI/gpt-neo-1.3B'  # 'EleutherAI/gpt-neo-1.3B' or 'EleutherAI/gpt-neo-2.7B'
	# ############################# Microsoft BERT Models ############################# #
	elif model == 'codebert-base' or model == 'codebert':
		huggingface_path = 'microsoft/codebert-base'  # 125M
	# ############################# Codesage Models ############################# #
	elif model == 'codesage-base':
		huggingface_path = 'codesage/codesage-base'  # 365M
	elif model == 'codesage-small':
		huggingface_path = 'codesage/codesage-small'  # 130M
	elif model == 'codesage-large':
		huggingface_path = 'codesage/codesage-large'  # 1.3B
	# ############################# Microsoft Phi Models ############################## #
	elif model == 'phi-1':
		huggingface_path = 'microsoft/phi-1'  # 1.3B
	elif model == 'phi-1_5':
		huggingface_path = 'microsoft/phi-1_5'  # 1.3B + augmented with a new data source (NLP synthetic texts)
	elif model == 'phi-2':
		huggingface_path = 'microsoft/phi-2'  # 2.7B + augmented with new data sources (NLP synthetic texts + websites)
	elif model == 'phi-3':
		huggingface_path = 'microsoft/Phi-3-mini-4k-instruct'  # 3.8B + larger and more advanced versions of the datasets used in phi-2. Also have 128k version
	elif model == 'phi-3-small':
		huggingface_path = 'microsoft/Phi-3-small-8k-instruct'  # 7B. Also have 128k version
	elif model == 'phi-3-medium':
		huggingface_path = 'microsoft/Phi-3-medium-4k-instruct'  # 14B. Also have 128k version
	# ############################# Salesforce CodeT5 Models ############################# #
	elif model == 'codet5-base':
		huggingface_path = 'Salesforce/codet5-base'  # (220M)
	elif model == 'codet5-large':
		huggingface_path = 'Salesforce/codet5-large'  # (770M) Can use codet5-large-ntp-py for Python, else codet5-large
	elif model == 'codet5p-110m-embedding':
		huggingface_path = 'Salesforce/codet5p-110m-embedding'
	elif model == 'codet5p-220m':
		huggingface_path = 'Salesforce/codet5p-220m-py'  # Can use codet5p-220m-py for Python, else codet5p-220m
	elif model == 'codet5p-770m':
		huggingface_path = 'Salesforce/codet5p-770m-py'  # Can use codet5p-770m-py for Python, else codet5p-770m
	elif model == 'codet5p-2b':
		huggingface_path = 'Salesforce/codet5p-2b'
	elif model == 'codet5p-6b':
		huggingface_path = 'Salesforce/codet5p-6b'
	elif model == 'codet5p-16b':
		huggingface_path = 'Salesforce/codet5p-16b'  # Also has `instructcodet5p-16b`
	# ############################# Salesforce CodeGen Models ############################# #
	elif model == 'codegen-350M':
		huggingface_path = 'Salesforce/codegen-350M-mono'  # Can use mono for Python, else multi
	elif model == 'codegen-2B':
		huggingface_path = 'Salesforce/codegen-2B-mono'  # Can use mono for Python, else multi
	elif model == 'codegen-6B':
		huggingface_path = 'Salesforce/codegen-6B-mono'  # Can use mono for Python, else multi
	elif model == 'codegen-16B':
		huggingface_path = 'Salesforce/codegen-16B-mono'  # Can use mono for Python, else multi
	elif model == 'codegen2-1B':
		huggingface_path = 'Salesforce/codegen2-1B'
	elif model == 'codegen2-3_7B':
		huggingface_path = 'Salesforce/codegen2-3_7B'
	elif model == 'codegen2-7B':
		huggingface_path = 'Salesforce/codegen2-7B'
	elif model == 'codegen2-16B':
		huggingface_path = 'Salesforce/codegen2-16B'
	elif model == 'codegen25-7b-multi':
		huggingface_path = 'Salesforce/codegen25-7b-mono'  # Can use mono for Python, else multi. Also has `-instruct`
	# ############################# Meta LLama Models ############################# #
	elif model == 'CodeLlama-7b-Python-hf':
		huggingface_path = 'codellama/CodeLlama-7b-Python-hf'
	elif model == 'CodeLlama-13b-Python-hf':
		huggingface_path = 'codellama/CodeLlama-13b-Python-hf'
	elif model == 'CodeLlama-34b-Python-hf':
		huggingface_path = 'codellama/CodeLlama-34b-Python-hf'  # 70b is only provided when requested
	elif model == 'Meta-Llama-3-8B':
		huggingface_path = 'meta-llama/Meta-Llama-3-8B'  # Also have `Meta-Llama-3-8B-instruct`
	elif model == 'Meta-Llama-3-70B':
		huggingface_path = 'meta-llama/Meta-Llama-3-70B'
	# ############################# DeepSeek-Coder Models ############################# #
	elif model == 'deepseek-coder-1.3b-base':
		huggingface_path = 'deepseek-ai/deepseek-coder-1.3b-base'
	elif model == 'deepseek-coder-7b-base':
		huggingface_path = 'deepseek-ai/deepseek-coder-7b-base-v1.5'  # This 1.5version is of size 6.9B which is further trained to be better than 6.7B
	elif model == 'deepseek-coder-33b-base':
		huggingface_path = 'deepseek-ai/deepseek-coder-33b-base'
	else:
		raise NotImplementedError()
	
	return huggingface_path
