#!/bin/bash
#SBATCH --partition=nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --qos=high
#SBATCH --job-name=llama_ab
#SBATCH --output=%j_output.log
#SBATCH --exclude=erc-hpc-comp053

source ~/miniconda3/etc/profile.d/conda.sh
# Activate the conda environment
conda activate openr1
# 切换到工作目录
cd /scratch/prj/inf_rate/qinglin/llama_decoding/decoding_token_position/

gpuid=1
datasets=("gsm8k" "gsm_hard" "svamp" "strategyqa")
#use_abs_list=(0 1)
repeat=(2 3)
model_name="meta-llama/Meta-Llama-3-8B-Instruct"

for repeat_num in "${repeat[@]}"; do
#  for use_abs in "${use_abs_list[@]}"; do
  for dataset in "${datasets[@]}"; do
  # 在这里循环 dimension 可以取 20 或 50
    for dimension in 50; do
      # 再循环 shot 值
      for shot in 0 8; do

        output_folder="./model_logs/llama/function_ei_final/"
        mkdir -p "$output_folder"

        # 打印任务开始时间和信息
        echo "Starting inference for model=${model_name}, dataset=${dataset}, shot=${shot}, dimension=${dimension}, num_repeats=${repeat_num} at $(date)"

        # 运行Python脚本，保存输出到日志文件
        CUDA_VISIBLE_DEVICES=$gpuid nohup python inference_mcts_postion.py \
          --model_name "$model_name" \
          --dataset "$dataset" \
          --begin 0 \
          --end 250 \
          --continue_predict 0 \
          --random_seed "$repeat" \
          --num_repeats 5 \
          --num_iteration 1 \
          --add_best_node 1 \
          --depth_limit 5 \

          --num_tokens 1 \
          --stop_threshold 0.01 \
          --shot "$shot" \
          --dimention "$dimension" \
          --function_method "ei" \
          --output_base_path ./output \
          > "$output_folder/$(date +"%m%d_%H%M%S")_${dataset}_shot_${shot}_function_pi_repeat${repeat_num}.log" 2>&1

      done
      python ./utils/finish_push.py "Finished function_pi for dimension=${dimension}, model=${model_name}, dataset=${dataset}"
    done
#    done
  done
done