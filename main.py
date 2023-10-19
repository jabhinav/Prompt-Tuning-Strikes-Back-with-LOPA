import argparse
import json
import multiprocessing
import os
from collections import OrderedDict
from datetime import datetime

import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup
from transformers.modeling_utils import unwrap_model

import logging
from custom_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from utils.data import LibrarySampleDataset, APPSBaseDataset
from utils.model import load_tokenizer, load_base_model, get_huggingface_path

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True, mode=0o777)

logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("\n\n# ################# Learning Libraries ################# #\n\n")


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    log_p = F.log_softmax(logits, dim=2)
    logpy = torch.gather(log_p, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def get_response_log_probs(args, batch, tokenizer, model, library_idx):
    
    prompt, prompt_mask, response, response_mask = batch
    batch_size = prompt.size(0)
    
    resp_logits = model(
        input_ids=prompt, attention_mask=prompt_mask,
        labels=response, output_hidden_states=True, library_idx=library_idx
    )['logits']
    
    # # Append labels [=-100] for the latent prompt to the response
    prefix = torch.full((batch_size, args.num_virtual_tokens), -100).to(response.device)
    response = torch.cat((prefix, response), dim=1)
    # # Append response_mask with 0s for the latent prompt (not the mask for attending to latent prompt)
    prefix_resp_mask = torch.zeros((batch_size, args.num_virtual_tokens)).to(response_mask.device)
    response_mask = torch.cat((prefix_resp_mask, response_mask), dim=1)
    
    response[response == -100] = tokenizer.pad_token_id  # Replace -100 with pad_token_id
    resp_labels = response.contiguous()
    
    resp_log_prob = logprobs_from_logits(resp_logits, resp_labels)
    resp_log_prob = resp_log_prob * response_mask
    
    # Likelihood of the sample coming from the latent prompt of library k
    resp_log_prob = resp_log_prob.sum(dim=1)
    
    return resp_log_prob


@torch.no_grad()
def compute_responsibilities(args, batch, tokenizer, model) -> torch.Tensor:
    
    """
    Compute the responsibilities i.e. posterior probabilities of the sample coming from the latent prompt of each
    library.
    :param args:
    :param batch:
    :param tokenizer:
    :param model:
    :return:
    """
    
    batch_size = batch[0].size(0)
    
    # Create a tensor of shape (n_samples, num_libraries) to store the responsibilities
    likelihood = torch.zeros((batch_size, args.num_libraries)).to(args.device)
    
    for k in range(args.num_libraries):
        
        # Store the likelihood of the sample coming from the latent prompt of library k
        likelihood[:, k] = get_response_log_probs(args, batch, tokenizer, model, k)
    
    # Normalize the responsibilities (prior can be uniform, thus cancelled out)
    responsibilities = F.softmax(likelihood, dim=1)
    
    return responsibilities.detach()


def learn(args):
    # Initialise wandb
    if args.wandb_logging:
        wandb.init(project=args.project_name, config=vars(args))
    
    # Get the tokenizer
    tokenizer = load_tokenizer(args.model_type, args.tokenizer_name)
    
    # Get the config
    peft_config = PromptTuningConfig(
        task_type=TaskType.MULTI_CAUSAL_LM,  # CAUSAL_LM, SEQ_2_SEQ_LM for Dec-only, Enc-Dec. MULTI is my custom field.
        prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
        num_virtual_tokens=args.num_virtual_tokens,
        # prompt_tuning_init_text="Classify if tweet is a complaint or not:",  # Use this if prompt_tuning_init is TEXT
        # tokenizer_name_or_path=args.model_name_or_path,  # Use this if prompt_tuning_init is TEXT
        num_init_clusters=args.num_libraries,  # My custom field
    )
    
    # Get the dataset
    dataset = LibrarySampleDataset(
        path_to_data=args.path_to_data,
        tokenizer=tokenizer,
        max_length=args.max_target_length
    )
    
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # Prepare training data loader
    sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    args.num_training_steps = (len(train_dataloader) * args.num_iters)
    
    # Get the model
    _, model = load_base_model(
        model_type=args.model_type,
        config_name=args.config_name,
        model_path=args.model_name_or_path
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Prompt tuning: embedding_dim * num_virtual_tokens * num_libraries
    
    # Get the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
    )
    
    # GPU-ize the model
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info("Starting EM for Library Learning")
    
    # # Debug Load the model
    # model.load_adapter(model_id=args.save_at, adapter_name='default')
    
    # ######################################### Initialisation for EM ############################################## #
    # Initialise the model parameters i.e. latent prompt embeddings for each library
    # This is equivalent to latching each library to a random sample from the dataset
    if args.pre_num_iters > 0:
        rdm_idxs = torch.randint(0, len(dataset), (args.num_libraries,))
        for k in range(args.num_libraries):
    
            logger.info("Initialisation for Library %d", k)
            for i in tqdm(range(args.pre_num_iters), desc=f"Init. Iterations Lib {k}", position=0, leave=True):
                batch = dataset.sample(rdm_idxs[k])
                batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
    
                # Get the response log-probability of the sample coming from the latent prompt of library k
                resp_log_prob = get_response_log_probs(args, batch, tokenizer, model, k)
    
                loss = -resp_log_prob.sum()  # responsibility = 1 for the sample coming from the latent prompt of library k
    
                # Update the model parameters
                loss.backward()
                optimizer.step()
                lr_scheduler.step()  # Make sure this is constant schedule with no warmup
                optimizer.zero_grad()
    
                logger.info(f"Iter {i} Loss: {loss.detach().cpu().numpy().item()}")
    
    # ################################################## EM ####################################################### #
    # Let's do EM to update the model with prompt-tuning
    for i in tqdm(range(args.num_iters), desc="EM Iterations", position=0, leave=True):
        
        # ############################################### E-Step #################################################### #
        # E-Step: Compute responsibilities corresponding to each program coming from some latent prompt of a library
        batch = next(iter(train_dataloader))
        batch = tuple(t.to(args.device) for t in batch)
        
        # Posterior probabilities of the sample coming from the latent prompt of each library := p(z_k|x_n)
        responsibilities = compute_responsibilities(args, batch, tokenizer, model)
        
        # To prevent underflow, clip the responsibilities to a minimum value
        responsibilities = responsibilities.clamp(min=1e-8)
        
        # ############################################### M-Step #################################################### #
        # M-Step: Update the model parameters i.e. latent prompt embeddings for each library
        #         by maximizing the likelihood of the data coming from the latent prompt of the library
        
        q_func = 0  # Total log-likelihood of the data coming from library, metric to track convergence
        responsibilities.to(args.device)
        
        # Library Book-keeping
        lib_train_logs = {}
        for k in range(args.num_libraries):
            
            # Likelihood of the sample coming from the latent prompt of library := p(x_n|z_k)
            resp_log_prob = get_response_log_probs(args, batch, tokenizer, model, k)
            
            # Re-normalise the responsibilities for library k -> Avoids numerical instability and does not affect EM
            norm_responsibilities = responsibilities[:, k] / responsibilities[:, k].sum()
            
            # Check norm_responsibilities are non-zero
            try:
                assert (norm_responsibilities != 0).all()
            except AssertionError:
                logger.info(f"Some responsibilities (after norm) for library {k} are still zero = {norm_responsibilities}")
            
            # Compute Loss = Negative Log Likelihood of the sample coming from the latent prompt of library k
            loss = -(resp_log_prob * norm_responsibilities.detach()).sum()
            
            # Compute the gradient norm
            # grad_norm = compute_grad_norm(model)
            
            # Update the model parameters
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update the total log-likelihood of the data coming from library
            q_func += -loss.detach().cpu().numpy().item()
            
            # Bookkeeping
            lib_train_logs[f"loss/lib_{k}"] = loss.detach().cpu().numpy().item()
        
        logger.info("Iteration: %d, Q-Func: %.4f", i, q_func)
        
        if args.wandb_logging:
            lib_train_logs.update({'q_func': q_func})
            wandb.log(lib_train_logs, step=i)
        
    # ################################################ Save Model ################################################## #
    # Save the model (by saving the trained embeddings for the latent prompt of each library)
    model = unwrap_model(model)
    logger.info("Saving the model at: %s", args.save_at)
    model.save_pretrained(save_directory=args.save_at)
    
    # ####################################### Compute final responsibilities ####################################### #
    # Debug by showing the responsibilities of each sample
    logger.info("\n\n# ################# Responsibilities ################# #")
    for i in tqdm(range(len(dataset)), desc="Computing Final Responsibilities", position=0, leave=True):
        
        batch = dataset.sample(i)
        batch = tuple(t.unsqueeze(0).to(args.device) for t in batch)
        
        responsibilities = compute_responsibilities(args, batch, tokenizer, model)
        # Debug by showing the responsibilities of each sample
        logger.info(f"[Responsibilities] {dataset.ids[i]}: {responsibilities.cpu().numpy()[0].tolist()}")
        

def get_config():
    data_type = "libraryCustom"
    model_type = "codegen2-1B"  # codegen2-1B, codegen-350M, CodeLlama-7b-Python-hf, codegen2-3_7B
    huggingface_path = get_huggingface_path(model_type)
    
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--wandb_logging', type=bool, default=True)
    parser.add_argument('--project_name', type=str, default='PromptTuningModel')
    parser.add_argument('--path_to_data', type=str, default='./sample_codes_processed')
    parser.add_argument('--save_at', type=str, default=log_dir + '/PromptTuningMultiModel')
    
    # Prompt Tuning
    parser.add_argument('--max_prefix_length', type=int, default=0)
    parser.add_argument('--num_virtual_tokens', type=int, default=20)
    parser.add_argument('--max_target_length', type=int, default=512)  # 512
    parser.add_argument('--num_libraries', type=int, default=4)
    
    # Policy Model
    parser.add_argument("--model_type", default=model_type, type=str)
    parser.add_argument("--model_name_or_path", type=str, default=huggingface_path)
    parser.add_argument("--config_name", type=str, default=huggingface_path)
    parser.add_argument("--tokenizer_name", type=str, default=huggingface_path)
    
    # Training
    parser.add_argument("--num_iters", default=50, type=int)
    parser.add_argument("--pre_num_iters", default=10, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int)
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    # Other
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Effective max target length for program generation
    args.max_target_length = args.max_target_length - args.num_virtual_tokens
    
    # Set distributed training parameters
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    cpu_cont = multiprocessing.cpu_count()  # Gives number of logical CPU cores
    # Recommended reading: https://superfastpython.com/multiprocessing-pool-num-workers/
    args.cpu_cont = cpu_cont - int(cpu_cont / 2)  # Ignore half of the cores
    
    # Initial device allocation for rl-based training [will be used by RL only]
    args.model_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count(using): %d, "
                   "cpu count(available): %d", args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1),
                   args.cpu_cont, cpu_cont)
    
    # Log the config
    config: dict = vars(args)
    config = {key: str(value) for key, value in config.items()}
    config = OrderedDict(sorted(config.items()))
    logger.info(json.dumps(config, indent=4))
    
    return args
    

if __name__ == '__main__':
    _args = get_config()
    learn(_args)
