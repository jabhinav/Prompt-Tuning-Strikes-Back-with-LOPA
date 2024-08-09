#!/bin/bash

# Define the list of lp_rank and num_tokens
lp_rank_list=(1)
virtual_tokens_list=(50)
tuning_method="lopa"
fm="gpt2-medium"
categories=("seen" "unseen" "all")

 for m in "${virtual_tokens_list[@]}"; do
    # Iterate over each combination
    for lp_rank in "${lp_rank_list[@]}"; do
          # Create a unique directory name
          log_dir="logging/webnlg_lopa_r${lp_rank}_m${m}"

          # ##################################### Train the model ################################################# #
          accelerate launch --config_file config_files/config_ds_zero_stage2_no_fp16.yaml tune_foundation_model.py --peft_method "$tuning_method" --task_name nlg_webnlg --model_type "$fm" --log_dir "$log_dir" --lp_rank "$lp_rank" --num_virtual_tokens "$m" --wandb_logging

          # ############################################# Get predictions ######################################### #
          result_dir="$log_dir/output"
          mkdir -p "$result_dir"

          load_adapter_from="$log_dir/final/PEFT"
          clf_predictor_path="$log_dir/final/clf_predictor.pt"
          accelerate launch generate_preds.py --peft_method "$tuning_method" --task_name nlg_webnlg --model_type "$fm" --lp_rank "$lp_rank"  --num_virtual_tokens "$m" --load_adapter_from "$load_adapter_from" --clf_predictor_path "$clf_predictor_path" --log_dir "$result_dir"

          # ############################################# Run the metrics ######################################## #
          cd nlgeval/GenerationEval/
          pred_dir="../../$result_dir"
          for cate in "${categories[@]}"; do
            references_path="$pred_dir/$cate/references_webnlg/reference"
            generations_path="$pred_dir/$cate/hypothesis_webnlg"
            results_path="$pred_dir/$cate/results.txt"

            python eval.py -R "$references_path" -H "$generations_path" -nr 6 -m bleu,ter > "$results_path"
          done
          cd ../..
    done
done
