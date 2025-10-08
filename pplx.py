import torch
from transformers import BertTokenizer, BertForMaskedLM
import math

# Load pre-trained model and tokenizer
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()

def calculate_pseudo_perplexity(text):
    """
    Calculates the pseudo-perplexity of a text using a BERT Masked Language Model.
    """
    # Tokenize the text
    tokenize_input = tokenizer.tokenize(text)
    
    # Add CLS and SEP tokens
    # Note: BERT models require special tokens at the beginning and end
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    
    # We don't want to mask the special tokens [CLS] and [SEP]
    # In this simple example, we are not batching, so we'll just slice
    # A more robust implementation would handle batches and padding
    if tensor_input.size(1) < 3:
      print("Sentence is too short to calculate pseudo-perplexity.")
      return float('inf')
      
    sent_loss = 0.
    
    # Iterate through each token in the sentence to be masked
    # We skip the first ([CLS]) and last ([SEP]) tokens
    for i, token_id in enumerate(tensor_input[0][1:-1], 1):
        
        # Create a copy of the input tensor
        masked_input = tensor_input.clone()
        
        # Mask the token at the current position
        masked_input[0][i] = tokenizer.mask_token_id
        
        with torch.no_grad():
            output = model(masked_input)
            
        # Get the logits for the masked token
        logits = output.logits[0, i, :]
        
        # Get the log probability of the original token
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        token_log_prob = log_probs[token_id].item()
        
        # Add to sentence loss (negative log-likelihood)
        sent_loss -= token_log_prob

    # The final score is the exponent of the average loss
    pseudo_ppl = math.exp(sent_loss / (len(tokenize_input)))
    return pseudo_ppl

# --- Example Usage ---
string1 = "The quick brown fox jumps over the lazy dog."
string3 = "Jumps dog over fox brown quick the lazy."

print(f"'{string1}'")
print(f"Pseudo-Perplexity: {calculate_pseudo_perplexity(string1):.2f}\n")

print(f"'{string3}'")
print(f"Pseudo-Perplexity: {calculate_pseudo_perplexity(string3):.2f}\n")
