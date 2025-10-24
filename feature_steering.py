# Standard imports
import random
import os
import math
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from vllm import LLM, SamplingParams
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import re
import torch.nn.functional as F
from functools import partial
from sae_lens import SAE, HookedSAETransformer
import argparse
import inspect
from collections import Counter
from utils.process_answer import parse_llm_response

torch.set_grad_enabled(False)

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

# def compute_metrics(input_sentence, metric):
#     if metric == 'rs':
#         return metrics.n_gram.calculate_rep_n(input_sentence, 1)
#     if metric == 'ie':
#         return metrics.Information_Entropy.calculate_ngram_entropy(input_sentence, 1)

# def access_dataset(dataset):
#     input_list = []
#     ds = load_dataset(dataset)
#     if 'Academic' in dataset:
#         # dataset="DisgustingOzil/Academic_dataset_ShortQA"
#         for i in range(1): 
#             response = ds['train']['response'][i]
            
#             question = re.search(r'<question>(.*?)</question>', response)
#             question = question.group(1).strip() 
#             input_list.append(question)
#         return input_list

#     if 'natural' in dataset:
#         # dataset = "sentence-transformers/natural-questions"
#         for i in range(1):
#             question = ds['train']['query'][i]
#             input_list.append(question)
#         return input_list
#     if 'Diversity' in dataset:
#         for i in range(10):
#             question = ds['train']['question'][i]
#             input_list.append(question)
#         return input_list

# def steering_hook(
#     activations,
#     hook,
#     sae,
#     latent_idxs,  
#     steering_coefficient
#     ):
#     """
#     Steers the model by returning a modified activations tensor, with multiples of the steering vectors added 
#     to all sequence positions for each specified latent index.
#     """
#     # give steering vector to activations
#     for latent_idx in latent_idxs:
#         activations += steering_coefficient * sae.W_dec[latent_idx]
    
#     return activations

# def my_steering_hook(
#     weight,
#     activations,
#     hook,
#     sae,
#     latent_idxs,  
#     ):
#     """
#     Steers the model by returning a modified activations tensor, with multiples of the steering vectors added 
#     to all sequence positions for each specified latent index.
#     """
#     original_norm = torch.norm(activations)
#     print('original norm', original_norm)
#     # give steering vector to activations
#     for w, latent_idx in zip(weight, latent_idxs) :
#         activations += w * sae.W_dec[latent_idx]
#     new_norm = torch.norm(activations)
#     print('new norm', new_norm)
#     activations = activations * (original_norm / rnew_norm_before)
#     return activations

# def generate_with_steering(
#     model,
#     tokenizer,
#     sae,
#     prompt,
#     latent_idxs, 
#     steering_coefficient,
#     max_new_tokens):
#     """
#     Generates text with steering. A multiple of the steering vector (the decoder weight for the specified latents)
#     is added to the last sequence position before every forward pass for each latent index.
#     """
#     _steering_hook = partial(
#         steering_hook,
#         sae=sae,
#         latent_idxs=latent_idxs,  
#         steering_coefficient=steering_coefficient,
#     )
#     GENERATE_KWARGS = dict(temperature=0.5,freq_penalty=2.0, verbose=False)
#     with model.hooks(fwd_hooks=[(sae.hook_name, _steering_hook)]):
#         output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)
#         print(output)
#         sys.exit()

#     output_edited = output.split("\n", 1)[-1].strip()
#     return output_edited

# def my_generate_with_steering(
#     weights,
#     model: HookedSAETransformer,
#     tokenizer,
#     sae: SAE,
#     prompt: str,
#     latent_idxs: list[int], 
#     generation_args):
#     """
#     Generates text with steering. A multiple of the steering vector (the decoder weight for the specified latents)
#     is added to the last sequence position before every forward pass for each latent index.
#     """
#     outputs = []
#     for weight in weights:
            
#         _steering_hook = partial(
#             my_steering_hook,
#             weight, 
#             sae=sae,
#             latent_idxs=latent_idxs,  
#         )
#         GENERATE_KWARGS = dict(temperature=0.5,freq_penalty=2.0, verbose=False)
#         with model.hooks(fwd_hooks=(_steering_hook)): # fwd_hooks=[(sae.hook_name, _steering_hook)]
#             output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

#         output_edited = output.split("\n", 1)[-1].strip()
#         outputs.append(output_edited)

#     return outputs

# def duc(model_path, dataset, save_path):
#     # Model Preparation
#     if 'gpt' in model_path:
#         llm = HookedSAETransformer.from_pretrained('gpt2', device = device)
#         tokenizer = AutoTokenizer.from_pretrained('gpt2')
#         sae, cfg_dict, sparsity = SAE.from_pretrained(
#             release = "gpt2-small-res-jb", # <- Release name
#             sae_id = "blocks.9.hook_resid_pre", # <- SAE id (not always a hook point!)
#             device = device
#         )

#     elif 'gemma' in model_path:
#         llm = HookedSAETransformer.from_pretrained_no_processing('google/gemma-2-2b', torch_dtype=torch.float16, device = device)
#         tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
#         sae, cfg_dict, sparsity = SAE.from_pretrained(
#             release="gemma-scope-2b-pt-res-canonical",  # <- Release name
#             sae_id="layer_25/width_16k/canonical",  # <- SAE id (not always a hook point!)
#             device=device
#         )

#     elif 'llama' in model_path:
#         llm = HookedSAETransformer.from_pretrained('meta-llama/Llama-3.1-8B',  torch_dtype=torch.float16, device = device)
#         tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
#         sae, cfg_dict, sparsity = SAE.from_pretrained(
#             release="llama_scope_lxr_8x", 
#             sae_id="l24r_8x", 
#             device=device)

#     input_list = access_dataset(dataset)


#     if 'gpt' in model_path:
#         latent_idxs = [22275, 6972, 8357, 3615, 13944, 7798, 10178, 22317, 18380, 16631, 3661, 16888, 3164, 6371, 17597, 16894, 12873, 7083, 5295, 8848, 17443, 23990, 18929, 21963, 15147, 10931, 4051, 4025, 20200, 186, 19336, 15875, 7699, 5051, 7770, 24312]
#     elif 'gemma' in model_path:
#         latent_idxs = [3350, 2424, 2752, 11566, 11653, 3050, 11018, 1563, 3996, 13589, 7644, 15662, 13000, 13032, 2210, 15312, 12056, 2205, 13513, 94, 421, 3858, 4884, 12653, 10243, 5263, 6608, 9423, 10860, 11592, 7637, 7618, 14613, 8065, 8509, 7341, 2645, 15954, 1988, 5490, 11985, 16300, 4017, 11076, 5425, 11049, 5429, 9227, 9795, 10178, 10566, 11073, 13907, 16094, 4946, 6129, 630, 7543, 1883, 8280, 14727, 12656, 12493, 7704, 13775, 1008, 6206, 7624, 11423, 14848, 14950, 2678, 3440, 4051, 7827, 8575, 13593, 16186, 13586, 16105, 6789, 147, 514, 1000, 7470, 15037, 577, 1447, 1007, 2632, 4071, 4807, 5964, 6954, 10744, 12099, 12827, 14148, 1365, 6023]
#     elif 'llama' in model_path:
#         #latent_idxs = [12656,8575,7468,15812,10614,916,1866,10781,16247,640]
#         #latent_idxs = [26028,871,403,27783,507]
#         latent_idxs = [21185,25206,2152,26865,1108,17963]
#         #latent_idxs = [18296, 24765, 8472, 21781, 16733, 11346, 3941, 29941, 31173, 26396, 25590, 29723, 25206, 32613, 20033, 18957, 9680, 15259, 20466, 29066, 7969, 29415, 24354, 24941, 18182, 19643, 29904, 12634, 32632, 22157, 2608, 6510, 10308, 27359, 29803, 26231, 7630, 27239, 27251, 11473, 12961, 22690, 32342, 1471, 17056, 25173, 26361, 30940, 30991, 5673, 22781, 25515, 10135, 3407, 3435, 4813, 28580, 25039, 32725, 403, 731, 1948, 26004, 20092, 496, 5184, 9001, 16612, 25450, 30824, 1046, 25758, 1031, 22382, 22478, 5676, 24792, 26802, 29227, 4859, 31317, 8594, 1306, 13115, 17596, 19992, 22959, 23440, 29599, 30366, 20417, 28457, 10817, 23993, 15632, 3002, 6412, 10235, 24323, 24741]


#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     with open(save_path, 'w', encoding='utf-8') as f_out:
#             item = "Where is the Vatican located in Rome?"
#             steered_output= generate_with_steering(
#                 llm,
#                 tokenizer,
#                 sae,
#                 item,
#                 latent_idxs,
#                 steering_coefficient=0.01,  # roughly 1.5-2x the latent's max activation gpt2small 40
#             )
#             steered_output = steered_output.replace("\n", "↵")     
#             repeat_score = compute_metrics(steered_output, 'rs')
#             information_entropy = compute_metrics(steered_output, 'ie')
#             result={
#                 "question": item,
#                 "output": steered_output,
#                 "repeat score": repeat_score,
#                 "information entropy": information_entropy,
#             }
#             f_out.write(json.dumps(result, ensure_ascii=False)+'\n')

# def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
#     if seed is not None:
#         torch.manual_seed(seed)

#     with model.hooks(fwd_hooks=fwd_hooks):
#         tokenized = model.to_tokens(prompt_batch)
#         result = model.generate(
#             stop_at_eos=False,  # avoids a bug on MPS
#             input=tokenized,
#             max_new_tokens=50,
#             do_sample=True,
#             **kwargs,
#         )
#     return result

# def run_generate(example_prompt):
#     model.reset_hooks()
#     editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
#     res = hooked_generate(
#         [example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs
#     )

#     # Print results, removing the ugly beginning of sequence token
#     res_str = model.to_string(res[:, 1:])
#     print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

# def generate_with_SAE_model(
#         sae_model, 
#         sae,
#         weights, 
#         model,
#         input,
#         temperature=0.8,
#         top_p=1,
#         top_k=50,
#         # repetition_penalty=1.1,
#         n=5,
#         max_tokens=200,
#         # logprobs=1,
#         stop=[],
#         insert_embedding=None,
#         model_name="llama",
#         special_token_id=128025, 
#         seed = None,

#     ):
#     print(weights)
#     # sys.exit()
#     GENERATE_KWARGS = dict(
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         eos_token_id = [9413],
#         stop_at_eos=True, 
#         max_new_tokens=max_tokens,
#         return_type='embeds'
#     )
#     latent_idxs = [  3017,  6586, 10550, 10150, 14342,  9618,  8043,   484,  8456, 13749,
#                     8911,  3168, 11655,  4120, 14037]
#     coeff = 300
#     def steering_hook(resid, hook):
        
#         SAE_vectors = sae.W_dec[latent_idxs]
        
#         for idx in range(resid.size()[0]):
        
#             weighted_SAE_vectors = weights[idx][0].to('cuda:1') @ SAE_vectors 
        
#             scaling_factor = torch.linalg.norm(resid[idx])/torch.linalg.norm(resid[idx]+(weighted_SAE_vectors))
#             resid[idx] = (resid[idx]+(weighted_SAE_vectors))*scaling_factor*coeff

#         return resid
#     if seed is not None:
#         torch.manual_seed(seed)
    
#     hook_point = f"blocks.{25}.hook_resid_post"
#     # sae_model.reset_hooks()
#     editing_hooks = [(hook_point, steering_hook)]
#     with sae_model.hooks(fwd_hooks=editing_hooks):
#         tokenized = sae_model.to_tokens(input)
#         output = sae_model.generate(
#             input=tokenized,
#             do_sample=True,
#             **GENERATE_KWARGS,
#         )
#     print(inspect.signature(sae_model.generate))
#     logits = output @ sae_model.W_U + sae_model.b_U  # [1, seq_len, vocab_size]
#     print(logits.size())
#     def logits_to_text_and_prob(
#         logits: torch.Tensor,
#         model: HookedTransformer
#         ) -> List[Tuple[str, float]]:
#         logits = logits.cpu()
#         probs = F.softmax(logits, dim=-1)

#         max_probs, token_ids = torch.max(probs, dim=-1)

#         seq_probs = torch.prod(max_probs, dim=-1)

#         texts = model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


#         probs = [pr.item() for pr in seq_probs]
#         variances = logits.var(dim=(1, 2))
#         print(variances)
#         min_val, max_val = variances.min(), variances.max()
#         normalized = (variances - min_val) / (max_val - min_val + 1e-8)
#         normalized = [v.item() for v in normalized]
#         return texts, normalized
#     return logits_to_text_and_prob(logits, sae_model)


def generate_with_SAE_model_v2(
        args, 
        sae_model, 
        sae,
        weights, 
        model,
        input,
        indices,
        temperature=0.8,
        top_p=1,
        top_k=50,
        n=5,
        max_tokens=200,
        stop=[],
        insert_embedding=None,
        model_name="llama",
        special_token_id=128025, 
        seed = None,
        

    ):
    if args.sae_base_model=='google/gemma-2-2b':
        stop_at = [[9413], [81435], [22804]]
    if args.sae_base_model=='meta-llama/Llama-3.1-8B':
        stop_at = [[14924], [128001]]
    if args.sae_base_model=='mistralai/Mistral-7B-Instruct-v0.1':
        stop_at = []
    GENERATE_KWARGS = dict(
        temperature=args.generation_temp, # temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id = stop_at, # [9413], this was for gemma # for llama [14924], [128001]
        stop_at_eos=True, 
        max_new_tokens=max_tokens,
        return_type='str',
        freq_penalty=0.3,
    )
    
    # latent_idxs = [  3017,  6586, 10550, 10150, 14342,  9618,  8043,   484,  8456, 13749,
    #                 8911,  3168, 11655,  4120, 14037]
    
    #this one is negative ones 
    # latent_idxs = [14325, 15298, 15920,  8657, 14651, 15645,  5299,  1689,  6481, 10572,
    #      9964, 14186,    39, 15820, 12028]
    # # this is for llama 3.1 - 31
    # latent_idxs = [ 51079,  65620,  97403,  82437,  72031, 111072, 108025, 107403, 106671,
    #     115855,  68092,  96301,   5103, 118333,  32333]
    
    # # this is for llama 3.1 - 28 
    # latent_idxs = [ 32632,  15130,  48831,  29266,  63922,  72831,  96414,  94689,  34397,
    #     125985,  28217,  58214,  24499,  55691,  49146]
    # print(weights)
    # static_weights = [ 476.3905, 297.6617, 267.0218, 227.5295, 187.9083, 186.5379, 286.9755,
    #             154.5167, 179.0570, 142.9292, 133.2771, 118.7599, 104.7732, 104.6720,
    #             93.7085]
    # static_weights = [weight/max(static_weights) for weight in static_weights]
    # print(static_weights)
    # input = [i.replace('<|reserved_special_token_20|>', '') for i in input]
    # latent 
    def patch_resid(resid, hook, steering, scale=320):
        adjusted = resid + steering * scale
        noramlization_factor = (torch.norm(resid, p=2))/(torch.norm(adjusted , p=2))
        resid = adjusted*noramlization_factor
        resid = adjusted
        return resid
    
    if seed is not None:
        torch.manual_seed(seed)

    SAE_vectors = sae[0].W_dec[indices]
    hook_point = sae[1]['hook_name'] # we need a translation here args.hook_point 
    tokenized = sae_model.to_tokens(input)
    # print(input[0])
    # print(SAE_vectors.dtype,  weights[0][0].to('cuda:1').dtype)
    weighted_SAE_vectors = torch.stack([
        weight[0].to(args.sae_device).to(SAE_vectors.dtype) @ SAE_vectors for weight in weights
    ]).unsqueeze(1)

    # weighted_SAE_vectors = torch.stack([
    #     torch.tensor(static_weights).to('cuda:1') @ SAE_vectors for _ in range(5)
    # ]).unsqueeze(1) 
    if args.steering_off==False:
        with sae_model.hooks([(hook_point, partial(patch_resid, steering=weighted_SAE_vectors, scale=args.steering_scale))]): # 15 12 is a good retio for gemma 2 2b 4 is good for llamma 
            output = sae_model.generate(   
                                        tokenized,
                                        do_sample=True,
                                        # loss_per_token=True,
                                        **GENERATE_KWARGS,
                                        )
    if args.steering_off==True:
        output = sae_model.generate(   
                                        tokenized,
                                        do_sample=True,
                                        # loss_per_token=True,
                                        **GENERATE_KWARGS,
                                        )
    # logits = output @ sae_model.W_U + sae_model.b_U 
    # print(sae_model.all_composition_scores('Q'))
    # print(output)
    parsed_output, _ = parse_llm_response(args, output)
    encodings = sae_model.to_tokens(parsed_output)
    input_ids = encodings.to(args.sae_device)
    with torch.no_grad():
        logits = sae_model(input_ids, loss_per_token=True)   
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = torch.log(token_probs).sum(dim=-1)
        seq_probs = torch.exp(seq_log_probs)
        perplexity = torch.exp(-seq_log_probs / input_ids.size(1))
        p_min, p_max = perplexity.min(), perplexity.max()
        normalized_perplexity = (perplexity - p_min) / (p_max - p_min + 1e-8)
    normalized_perplexity_list = normalized_perplexity.detach().cpu().tolist()
    return parsed_output, [1-item for item in normalized_perplexity_list]  
    # def logits_to_text_and_prob(
    #     logits: torch.Tensor,
    #     model: HookedTransformer
    #     ) -> List[Tuple[str, float]]:
    #     logits = logits.cpu()
    #     probs = F.softmax(logits, dim=-1)

    #     max_probs, token_ids = torch.max(probs, dim=-1)

    #     seq_probs = torch.prod(max_probs, dim=-1)

    #     texts = model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


    #     probs = [pr.item() for pr in seq_probs]
    #     variances = logits.var(dim=(1, 2))
    #     print(variances)
    #     min_val, max_val = variances.min(), variances.max()
    #     normalized = (variances - min_val) / (max_val - min_val + 1e-8)
    #     normalized = [v.item() for v in normalized]
    #     return texts, normalized
    # return logits_to_text_and_prob(logits, sae_model)
    # print(type(output), output.size())   
    # sys.exit()     
    # texts = [sae_model.tokenizer.decode(out) for out in output] 

    # probs = [1-(txt.count('<eos>')/(txt.count('<eos>')+len(txt.split(' ')))) for txt in texts] # gemma 
    # probs = [1-(txt.count('<|end_of_text|>')/(txt.count('<|end_of_text|>')+len(txt.split(' ')))) for txt in texts] # llama 
    # def most_frequent_word_ratio(text: str) -> float:
    #     words = text.split(' ')
    #     if words == []:
    #         return 0.0
        
    #     counts = Counter(words)
    #     most_common_count = max(counts.values())
    #     total_words = len(words)
    #     return most_common_count / total_words
    # probs = [1-most_frequent_word_ratio(txt) for txt in texts]
    # return texts, probs




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, choices=["gpt2", "meta-llama/Llama-3.1-8B","google/gemma-2-2b"])
    parser.add_argument('--dataset', type=str, choices=["DisgustingOzil/Academic_dataset_ShortQA", "YokyYao/Diversity_Challenge", "sentence-transformers/natural-questions"])
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    duc(model_path=args.model_path, dataset=args.dataset, save_path=args.save_path)

if __name__ == "__main__":
    main()
