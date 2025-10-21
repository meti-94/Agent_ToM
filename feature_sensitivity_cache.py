import os
os.environ['HF_HOME'] = '/mnt/datadisk/Mehdi'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm
import sys
import json
import hashlib
from datetime import datetime


# from sae_lens import SAE

# release = "mistral-7b-res-wg"
# sae_id = "blocks.24.hook_resid_pre"
# sae = SAE.from_pretrained(release, sae_id)[0]

def sanitize(s: str) -> str:
    """
    Keep it filesystem-safe for directory names only.
    Replace path separators and whitespace clusters.
    """
    return s.replace("/", "__").replace("\\", "__").strip()

def args_fingerprint(args) -> str:
    """
    Stable hash from the args that influence computed indices.
    If any of these change, we recompute (i.e., a different directory).
    """
    fields = [
        "base_model_name", "release", "hook_point", "dataset_name",
        "n", "k", "beta", "theta", "device"
    ]
    payload = {f: getattr(args, f) for f in fields}
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def build_cache_paths(args, cache_root: str):
    """
    Build the cache directory using combination of model/release/hook_point and
    the args fingerprint as a subdirectory. The filename is constant ('indices.json'),
    so filename does NOT play any role in cache identity.
    """
    # Static part: model/release/hook_point
    static_dir = os.path.join(
        cache_root,
        sanitize(args.base_model_name),
        sanitize(args.release),
        sanitize(args.hook_point),
    )

    # Dynamic part: args fingerprint (captures n, k, beta, theta, etc.)
    fingerprint = args_fingerprint(args)
    dir_path = os.path.join(static_dir, fingerprint)

    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Constant filename regardless of args
    file_path = os.path.join(dir_path, "indices.json")
    return dir_path, file_path


def save_indices_json(file_path: str, top_k_indices, top_k_scores=None, meta: dict = None):
    data = {
        "top_k_indices": [int(i) for i in top_k_indices],  # ensure plain ints for JSON
    }
    if top_k_scores is not None:
        data["top_k_scores"] = [float(s) for s in top_k_scores]
    if meta is not None:
        data["metadata"] = meta
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def load_indices_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def main(args):
    # Prepare cache location
    cache_dir, cache_file = build_cache_paths(args, args.cache_root)

    # If cache exists and not forcing recompute, load and return
    if (not args.force_recompute) and os.path.isfile(cache_file):
        cached = load_indices_json(cache_file)
        top_k_indices = cached.get("top_k_indices", [])
        print(f"[CACHE HIT] Loaded top-k indices from:\n  {cache_file}")
        print(top_k_indices)
        return

    # Otherwise compute indices
    print('Starting computation...')
    # train_dataset = load_dataset(args.dataset_name, "main", split="train", download_mode="force_redownload")
    train_dataset = load_dataset("json", data_files="data/my_data/gsm8k_train.json")
    train_dataset = train_dataset["train"]
    shuffled_dataset = train_dataset.shuffle(seed=42)
    random_1000_samples = shuffled_dataset.select(range(args.n))

    if 'gsm' in args.dataset_name:
        prefixes = [
            "As a highly qualified mathematics teacher, you excel at solving problems systematically and explaining solutions with clarity. I am your student, eager to learn. Please solve the following problem:",
            "As an excellent mathematics teacher, you always guide your students correctly through math problems. I am one of your students, eager to learn. Please answer the following question:",
            "As a respected mathematics professor with deep expertise in solving complex problems, you are known for your clarity and precision. I am your student and need help. Please solve the following question for me:",
            "As a world-renowned mathematics teacher, you are highly skilled at solving problems precisely and explaining them effectively. I am your student, struggling with a question. Please solve the following task for me:",
            "As a mathematics expert with strong problem-solving skills, you are deeply trusted by your students. I am one of them and need your help. Please solve the following problem for me:"
        ]
    else:
        # Fallback if dataset_name doesn't include 'gsm'
        prefixes = ["Please answer the following question:"]

    positive_samples = [f'{pre}\n{qst}' for pre in prefixes for qst in random_1000_samples['question']]
    negative_samples = [f'\n{qst}' for _ in prefixes for qst in random_1000_samples['question']]

    device = args.device

    # Load models
    model = HookedTransformer.from_pretrained(args.base_model_name, device=device, dtype=torch.float16)
    sae = SAE.from_pretrained(
        release=args.release,
        sae_id=args.hook_point,
        device=device,
    )

    dimension = sae[1]['d_sae']
    pos_feature_acts = torch.zeros(dimension, device=device)
    neg_feature_acts = torch.zeros(dimension, device=device)
    pos_delta = torch.zeros(dimension, device=device)
    neg_delta = torch.zeros(dimension, device=device)

    sae[0].eval()
    with torch.no_grad():
        total = args.n * len(prefixes)
        for idx, (pos, neg) in tqdm(enumerate(zip(positive_samples, negative_samples)), total=total, desc='Processing Samples ...'):
            tokens = model.to_tokens([pos, neg], prepend_bos=True)
            _, cache = model.run_with_cache(tokens)
            feature_acts = sae[0].encode(cache[sae[1]['hook_name']])[:, -1, :]

            pos_feature_acts = torch.add(pos_feature_acts, feature_acts[0, :])
            neg_feature_acts = torch.add(neg_feature_acts, feature_acts[1, :])

            # IMPORTANT: make sure to use ">" not "&gt;" in your code
            pos_delta = torch.add(pos_delta, (feature_acts[0, :] > args.theta).float())
            neg_delta = torch.add(neg_delta, (feature_acts[1, :] > args.theta).float())

    # Cleanup heavy objects
    del cache
    del sae
    del model

    weights = torch.div(torch.add(pos_feature_acts, neg_feature_acts), 2 * args.n)
    mu = torch.div(torch.sub(pos_feature_acts, neg_feature_acts), args.n)
    delta = torch.div(torch.subtract(pos_delta, neg_delta), args.n)

    sensitivity_scores = mu + args.beta * delta
    print(f"Selecting top-{args.k} features...")
    top_k_scores, top_k_indices = torch.topk(sensitivity_scores, args.k, largest=True)

    # Prepare JSON-safe outputs and metadata
    top_k_indices_list = top_k_indices.detach().cpu().tolist()
    top_k_scores_list = top_k_scores.detach().cpu().tolist()
    meta = {
        "base_model_name": args.base_model_name,
        "release": args.release,
        "hook_point": args.hook_point,
        "dataset_name": args.dataset_name,
        "n": args.n,
        "k": args.k,
        "beta": args.beta,
        "theta": args.theta,
        "device": args.device,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Save results to predefined path
    save_indices_json(cache_file, top_k_indices_list, top_k_scores_list, meta)
    print(f"[SAVED] top_k_indices saved to:\n  {cache_file}")
    print(top_k_indices_list)
    # (Optional) also print some weights for inspection
    print(weights[:10], weights.size(), weights[top_k_indices])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAE-based feature sensitivity analysis on GSM8K math dataset with caching."
    )
    parser.add_argument(
        "--beta", type=float, default=1.5,
        help="Weighting factor for delta when computing sensitivity scores (default: 1.5)."
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of top features to select (default: 10)."
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of samples."
    )
    parser.add_argument(
        "--theta", type=float, default=2.0,
        help="Activation threshold for counting feature deltas (default: 2.0)."
    )
    parser.add_argument(
        "--release", type=str, default="gpt2-small-res-jb",
        help="SAE release version (default: gpt2-small-res-jb)."
    )
    parser.add_argument(
        "--hook_point", type=str, default="blocks.8.hook_resid_pre",
        help="Model hook point for SAE extraction (default: blocks.8.hook_resid_pre)."
    )
    parser.add_argument(
        "--base_model_name", type=str, default="gpt2-small",
        help="Base model name to load (default: gpt2-small)."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="openai/gsm8k",
        help="Dataset to use for analysis (default: openai/gsm8k)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on: 'cpu', 'cuda', or 'mps' (default: cpu)."
    )
    parser.add_argument(
        "--cache_root", type=str, default="./cache",
        help="Root directory to store cached indices (default: ./cache)."
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore cache and recompute indices."
    )

    args = parser.parse_args()
    main(args)