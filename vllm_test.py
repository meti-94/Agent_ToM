import os
os.environ['HF_HOME'] = '/srv/scratch/CRUISE/Mehdi/HF'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from vllm import LLM
llm = LLM(model="Qwen/Qwen2-7B-Instruct", gpu_memory_utilization=0.5)