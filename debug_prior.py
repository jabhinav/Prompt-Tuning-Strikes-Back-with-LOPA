from utils.data import MBPP_Dataset_w_CodeBERT as CustomDataset
from utils.model import ClarificationCodeBERTPredictor
from utils.xformer import load_tokenizer
from utils.config import get_config
import torch
import numpy as np
from utils.xformer import get_huggingface_path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

args, _ = get_config()

# Add BERT specific args
args.bert_model_type = "codebert-base"
args.bert_tokenizer_name = get_huggingface_path(args.bert_model_type)
args.bert_model_name_or_path = get_huggingface_path(args.bert_model_type)
args.bert_config_name = get_huggingface_path(args.bert_model_type)
tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
dataset = CustomDataset(
			path_to_data=args.path_to_data,
			tokenizer=tokenizer,
			max_prompt_length=args.max_prompt_length,
			max_length=args.max_length,
			sample_problems=args.num_train_problems,
			mode='train'
		)
prior = ClarificationCodeBERTPredictor(args=args, output_dim=args.num_libraries)

loaded_state_dict = torch.load('./logging/new_predictor.pt', map_location="cpu")
loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
prior.load_state_dict(loaded_state_dict, strict=True)
del loaded_state_dict

# Store per clf pred probs to plot them later
clf_ = {
	str(idx): [] for idx in range(args.num_libraries)
}

pred_idxs = []
prior.eval()
with torch.no_grad():
	for idx in tqdm(range(len(dataset)), desc="Computing clf probs", total=len(dataset)):
		batch = dataset.sample(idx)
		batch = tuple(t.to(args.device) for t in batch)
		batch = tuple(torch.unsqueeze(t, 0) for t in batch)
		
		probs = prior(batch[0], batch[1])
		probs = probs.detach().cpu().numpy()[0]
		for k in range(args.num_libraries):
			clf_[str(k)].append(probs[k])
		
		pred_idxs.append(np.argmax(probs[0]).item())
		
		
		# print("Prior Input Seq: ", dataset.bert_tokenizer.decode(batch[0][0].detach().cpu().numpy()))
		# print("LM Input Seq: ", tokenizer.decode(batch[2][0].detach().cpu().numpy()))


print(pred_idxs)

# Plot the clf probs for each clf in a single plot
sns.set_theme(style="darkgrid")
for idx in range(args.num_libraries):
	plt.plot(clf_[str(idx)], label=f'clf_{idx}')
plt.legend()

plt.savefig(os.path.join(args.log_dir, 'clf_probs.png'))