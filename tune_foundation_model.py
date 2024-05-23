from accelerate.logging import MultiProcessAdapter

from utils.config import get_config
from utils.custom import is_rank_0


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	# Print a big message for peft method
	if is_rank_0():
		print("\n\n")
		print("#" * 100)
		print(f"PEFT Method: {args.peft_method}")
		print("#" * 100)
	
	if args.peft_method == 'lopa':
		from trainers.lopa import Trainer
		trainer = Trainer(args, logger)
	
	elif args.peft_method == 'lora':
		from trainers.lora import Trainer
		trainer = Trainer(args, logger)
	
	elif args.peft_method == 'pt':
		from trainers.pt import Trainer
		trainer = Trainer(args, logger)
	
	elif args.peft_method == 'idpg':
		from trainers.idpg import Trainer
		trainer = Trainer(args, logger)
	
	else:
		raise ValueError(f"PEFT method {args.peft_method} currently not supported.")

	trainer.train_loop()


if __name__ == '__main__':
	# To run with accelerate, $ accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml tune_lora_baseline.py
	main()