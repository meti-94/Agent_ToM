# Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration

[ICML 2025 Spotlight Paper](https://arxiv.org/abs/2505.24688)

## Installation

We recommend using **Python 3.11** and a clean Conda environment:

```bash
conda create -n soft_reasoning python=3.11.11
conda activate soft_reasoning
```

Then install the required packages via `pip`:


```bash
pip install torch==2.5.1 transformers==4.51.1 vllm==0.7.2 tqdm numpy pyyaml
```


## Quick Start

Run the script quickly with the following command:

```bash
python inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset gsm8k \
  --shot 8 \
  --begin 0 \
  --end 200
```

## Important Arguments

| Argument           | Description                                 |
| ------------------ |---------------------------------------------|
| `--model_name`     | Model name or path (e.g., Meta-Llama-3)     |
| `--dataset`        | Dataset name (`gsm8k`, `strategyqa`, etc.)  |
| `--shot`           | Number of few-shot examples (`0`=zero-shot) |



## Citation

```bibtex
@misc{zhu2025softreasoningnavigatingsolution,
  title={Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration},
  author={Qinglin Zhu and Runcong Zhao and Hanqi Yan and Yulan He and Yudong Chen and Lin Gui},
  year={2025},
  eprint={2505.24688},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.24688}
}
```
