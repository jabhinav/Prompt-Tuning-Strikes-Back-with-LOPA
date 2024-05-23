#!/bin/bash

# Define the list of lp_rank and num_tokens
model_list=("Meta-Llama-3-8B")  # Add your list of model names here
#model_list=("deepseek-coder-7b-base")  # Add your list of model names here
lp_rank_list=(4)

# Create overall_results.txt file
overall_results_file="logging/ablation/overall_results.txt"
echo "Model | LP_Rank | Metrics (Stage-1) | Metrics (Stage-2)" > "$overall_results_file"

for model in "${model_list[@]}"; do
    # Iterate over each combination
    for lp_rank in "${lp_rank_list[@]}"; do
          # # Create a unique directory name
          log_dir="logging/ablation/${model}_lp${lp_rank}_mbpp"

          # ##################################### Train the model ################################################# #
           # For others
          accelerate launch --config_file config_files/config_ds_zero_stage2_no_fp16.yaml tune_foundation_model.py --peft_method lopa --task_name mbpp --lp_rank "$lp_rank" --log_dir "$log_dir" --model_type "$model"

           # For FFT [Uncomment]
#          deepspeed tune_fft_baseline.py --path_to_ds_config ./zero_stage3_config.json --fp16 True --gradient_accumulation_steps 4  --peft_method fft --task_name mbpp --log_dir "$log_dir" --model_type "$model"

          # ############################################# Get predictions ######################################### #
          result_dir="$log_dir/results"
          mkdir -p "$result_dir"

          # For others
          load_adapter_from="$log_dir/final/PEFT"
          clf_predictor_path="$log_dir/final/clf_predictor.pt"
          accelerate launch generate_preds.py --peft_method lopa --task_name mbpp --lp_rank "$lp_rank" --log_dir "$result_dir" --load_adapter_from "$load_adapter_from" --clf_predictor_path "$clf_predictor_path" --model_type "$model"

          # For FFT [Uncomment]
#          load_base_from_path="$log_dir/PromptTuningMultiModel"
#          sharded_checkpoint_dir="$log_dir/PromptTuningMultiModel"
#          accelerate launch eval_mbpp_accelerated.py --log_dir "$result_dir" --sharded_checkpoint_dir "$sharded_checkpoint_dir" --model_type "$model"


          # #################################### Process predictions ############################################# #
          raw_generations_path="$result_dir/mbxp_solutions.json"
          python postprocess_mbpp_preds.py --path "$raw_generations_path"

          # ############################################# Run the metrics ######################################## #
          # Stage 1 post-processing
          tmp_metrics_file_stage1="logging/ablation/metrics_${model}_lp${lp_rank}_stage1.txt"
          generations_path="$result_dir/mbxp_solutions_post_processed_stage1.json"
          evaluate_functional_correctness "$generations_path" --problem_file mxeval/mbpp_test_release_v1.jsonl > "$tmp_metrics_file_stage1"

          # Stage 2 post-processing
          tmp_metrics_file_stage2="logging/ablation/metrics_${model}_lp${lp_rank}_stage2.txt"
          generations_path="$result_dir/mbxp_solutions_post_processed_stage2.json"
          evaluate_functional_correctness "$generations_path" --problem_file mxeval/mbpp_test_release_v1.jsonl > "$tmp_metrics_file_stage2"

          # Append the metrics to the overall_results.txt file
          echo "$model $lp_rank $(cat $tmp_metrics_file_stage1) $(cat $tmp_metrics_file_stage2)" >> "$overall_results_file"

          # Remove the temporary metrics file
          rm "$tmp_metrics_file_stage1"
          rm "$tmp_metrics_file_stage2"

    done
done
