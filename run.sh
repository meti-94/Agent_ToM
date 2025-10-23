#!/bin/bash

# Parameters
UCB_BETAS=(2.5 4.5 6.5)
STEERING_SCALES=(1 2 3 4)
FUNCTION_METHODS=("ei" "pi" "ucb")

# Common arguments
BASE_CMD="/home/mahdi/miniconda3/envs/soft_reasoning/bin/python inference.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --end 50 \
  --dataset gsm8k \
  --shot 4 \
  --dimention 15 \
  --cache_file /home/mahdi/Agent_ToM/cache/mistralai__Mistral-7B-Instruct-v0.1/mistral-7b-res-wg/blocks.24.hook_resid_pre/0d8af8be4b3194d9951e32db826a85eb2a5c0b95/indices.json \
  --sae_base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --release mistral-7b-res-wg \
  --hook_point blocks.24.hook_resid_pre"

# Output directory
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

# Run combinations
for beta in "${UCB_BETAS[@]}"; do
  for scale in "${STEERING_SCALES[@]}"; do
    for method in "${FUNCTION_METHODS[@]}"; do
      echo "Running with ucb_beta=$beta, steering_scale=$scale, function_method=$method"
      
      OUT_FILE="$OUTPUT_DIR/output_beta${beta}_scale${scale}_method${method}.txt"
      
      $BASE_CMD \
        --ucb_beta "$beta" \
        --function_method "$method" \
        --steering_scale "$scale" \
        > "$OUT_FILE" 2>&1
      
      echo "✅ Saved output to $OUT_FILE"
    done
  done
done
