from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

###Remember to explicity include <|endoftext|> in construction of strings

class LMReward:

    def __init__(self, model = "gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model.to(self.device)

    def str_likelihood(self, string):

        inputs = self.tokenizer(string, return_tensors="pt", padding = True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        #Remove prediction after <|endoftext|>  
        logits = outputs.logits[:, :-1, :] 
        
        #Shift labels, mask to the left to correspond to predictions
        labels = inputs['input_ids'][:, 1:]
        attention_masks = inputs['attention_mask'][:, 1:]  
        
        log_probs = F.log_softmax(logits, dim=-1)
        labels_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        str_log_probs = (attention_masks * labels_log_probs).sum(dim = 1)
        
        str_probs = torch.exp(str_log_probs)
        
        return str_probs, str_log_probs