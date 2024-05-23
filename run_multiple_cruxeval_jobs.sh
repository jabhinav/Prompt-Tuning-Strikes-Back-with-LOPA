#!/bin/bash

# Define the list of lp_rank and num_tokens
#enc_model_list=("codebert-base" "codesage-small" "codesage-base" "codesage-large")  # Add your list of model names here
model_list=("deepseek-coder-1.3b-base" "codegen-350M")  # Add your list of model names here: deepseek-coder-7b-base, Meta-Llama-3-8B
lp_rank_list=(4 2 1)
crux_tasks=("output" "input")

# Create overall_results.txt file
overall_results_file="logging/ablation/overall_results.txt"
echo "Model | LP_Rank | Task | Metrics" > "$overall_results_file"

for model in "${model_list[@]}"; do
    # Iterate over each combination
    for lp_rank in "${lp_rank_list[@]}"; do
        for task in "${crux_tasks[@]}"; do
            # Create a unique directory name
            log_dir="logging/ablation/${model}_lp${lp_rank}_${task}_task"

            # ##################################### Train the model ################################################# #
            # For others
            accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml tune_foundation_model.py --peft_method lopa --task_name "cruxeval_${task}_prediction" --lp_rank "$lp_rank" --log_dir "$log_dir" --model_type "$model"

            # For FFT [Uncomment]
#            deepspeed tune_fft_baseline.py --path_to_ds_config ./zero_stage2_config.json --fp16 True --gradient_accumulation_steps 1  --log_dir "$log_dir" --model_type "$model" --task_name "cruxeval_${task}_prediction"

            # ############################################# Get predictions ######################################### #
            result_dir="$log_dir/results"
            mkdir -p "$result_dir"

            # For others
            load_adapter_from="$log_dir/final/PEFT"
            clf_predictor_path="$log_dir/final/clf_predictor.pt"
            accelerate launch generate_preds.py --peft_method lopa --task_name "cruxeval_${task}_prediction" --lp_rank "$lp_rank" --log_dir "$result_dir" --load_adapter_from "$load_adapter_from" --clf_predictor_path "$clf_predictor_path" --model_type "$model" --cruxeval_task "${task}_prediction"

            # For FFT [Uncomment]
#            load_base_from_path="$log_dir/PromptTuningMultiModel/pretrained_model.pt"
#            sharded_checkpoint_dir="$log_dir/PromptTuningMultiModel"
#            accelerate launch eval_cruxeval_accelerated.py --log_dir "$result_dir" --sharded_checkpoint_dir "$sharded_checkpoint_dir" --model_type "$model" --cruxeval_task "${task}_prediction"

            # ################## Process predictions [Mine will overwrite the one in eval script] ################## #
            raw_generations_path="$result_dir/output_raw.json"
            python postprocess_cruxeval_preds.py --path "$raw_generations_path" --mode "$task"

            # ############################################# Run the metrics ######################################## #
            tmp_metrics_file="logging/ablation/metrics_${model}_lp${lp_rank}_task${task}.txt"
            generations_path="$result_dir/output.json"
            scored_results_path="$result_dir/output_scored.json"
            python cruxeval/evaluation/evaluate_generations.py --generations_path "$generations_path" --scored_results_path "$scored_results_path" --mode "$task" > "$tmp_metrics_file"

            # Append the metrics to the overall_results.txt file
            echo "$model $lp_rank $task $(cat $tmp_metrics_file)" >> "$overall_results_file"

            # Remove the temporary metrics file
            rm "$tmp_metrics_file"

        done
    done
done
