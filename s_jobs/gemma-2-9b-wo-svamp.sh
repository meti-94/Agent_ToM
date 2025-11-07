#!/bin/sh

#SBATCH --gres=gpu:2          # request 2 GPUs
#SBATCH --time=0
/bin/hostname

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"


# run python job native on a node
srun /home/unsw.mahdi/anaconda3/envs/agent_tom/bin/python \
     /home/unsw.mahdi/Agent_ToM/inference.py \
     --model_name google/gemma-2-9b \
     --dataset svamp \
     --end 50 \
     --shot 4 \
     --dimention 15 \
     --function_method ei \
     --cache_file '/home/unsw.mahdi/Agent_ToM/cache/google__gemma-2-9b/indices.json' \
     --freq_penalty 0.4 \
     --sae_base_model google/gemma-2-9b \
     --release gemma-scope-9b-pt-res-canonical \
     --hook_point layer_25/width_16k/canonical \
     --sae_device "cuda:1" \
     --device "cuda:0" \
     --steering_scale 12.0 \
     --sae_selection relevant \


