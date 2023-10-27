import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_model_id = "stevhliu/bloomz-560m_PROMPT_TUNING_CAUSAL_LM"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
	tokenizer.pad_token_id = tokenizer.eos_token_id

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

text_column = "Tweet text"
label_column = "text_label"
inputs = tokenizer(
	f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
	return_tensors="pt",
)

model.to(device)

with torch.no_grad():
	inputs = {k: v.to(device) for k, v in inputs.items()}
	outputs = model.generate(
		input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3,
		do_sample=True, top_p=0.9, top_k=0, temperature=0.9, num_return_sequences=1
	)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
