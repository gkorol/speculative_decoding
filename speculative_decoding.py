import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class DecisionGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        comp = torch.all(x == y)
        if comp:
            return True
        else:
            return False

class SpeculativeDecoder(torch.nn.Module):
    """
    A class implementing speculative decoding for language models.

    This class uses a larger target model and a smaller draft model to perform
    speculative decoding, potentially speeding up text generation.

    Attributes:
        device (str): The device to run the models on ('cuda' or 'cpu').
        target_model (AutoModelForCausalLM): The larger, more accurate language model.
        draft_model (AutoModelForCausalLM): The smaller, faster language model for draft predictions.
        tokenizer (AutoTokenizer): The tokenizer for both models.
    """

    def __init__(self, target_model_name, draft_model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the SpeculativeDecoder with target and draft models.

        Args:
            target_model_name (str): The name or path of the target (larger) model.
            draft_model_name (str): The name or path of the draft (smaller) model.
            device (str): The device to run the models on. Defaults to 'cuda' if available, else 'cpu'.
        """
        super().__init__()
        
        self.device = device
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(self.device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(self.device)
        self.gate = DecisionGate()
        # self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)        
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_model.eval()
        self.draft_model.eval()

    @staticmethod
    def sample(logits, temperature, top_k, top_p):
        """
        Adjust logits for sampling based on temperature, top-k, and top-p parameters.

        Args:
            logits (torch.Tensor): The input logits.
            temperature (float): The temperature for sampling.
            top_k (int): The number of top tokens to consider for top-k sampling.
            top_p (float): The cumulative probability threshold for top-p sampling.

        Returns:
            torch.Tensor: The adjusted probability distribution.
        """
        if temperature <= 1e-6:
            return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
        
        logits = logits / temperature
        
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return F.softmax(logits, dim=-1)

    # def forward(self, input_ids, pad_token_id, len_encoded_prompt, temperature=1.0, top_k=0, top_p=1.0, gamma=5, max_new_tokens=100):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # return self.generate(input_ids, pad_token_id, len_encoded_prompt, temperature, top_k, top_p, gamma, max_new_tokens)
        # return self.draft_model(input_ids)
        generated_tokens = input_ids.clone()  # Start with the input sequence
        max_new_tokens = 5 # TODO: this is a parameter

        for _ in range(max_new_tokens):
            # version 1)
            #   TODO: fix control flow error from if
            # with torch.no_grad():
            #     draft_logits = self.draft_model(generated_tokens).logits[:, -1, :]
            #     draft_tokens = torch.topk(draft_logits, k=1, dim=-1).indices  # Greedy selection
            #     candidate_sequence = torch.cat([generated_tokens, draft_tokens], dim=-1)
            #     # Target model evaluates the candidate sequence
            #     target_logits = self.target_model(candidate_sequence).logits[:, -len(draft_tokens):, :]
            #     target_tokens = torch.argmax(target_logits, dim=-1)  # Greedy decoding
            # # Compare draft tokens and target tokens
            # # if torch.equal(draft_tokens, target_tokens): # gives unsupported op (equal)
            # if torch.all(draft_tokens == target_tokens):
            #     # If they match, accept the draft tokens
            #     generated_tokens = candidate_sequence
            # else:
            #     # If mismatch, take only the first token from the target model
            #     generated_tokens = torch.cat([generated_tokens, target_tokens[:, :1]], dim=-1)
            # # Stop if EOS token is generated
            # if generated_tokens[0, -1] == 50256:  # GPT-2 EOS token
            #     break
            # end of version 1

            # version 2)
            #   uses the indices to check draft and target tokens...
            #   TODO fix removal of EOS
            #      or always do it as a post-processing step outside the model
            with torch.no_grad():
                draft_logits: torch.Tensor = self.draft_model(generated_tokens).logits  # (1, seq_len, vocab_size)
                draft_token: torch.Tensor = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)  # (1, 1)
                candidate_sequence: torch.Tensor = torch.cat([generated_tokens, draft_token], dim=-1)  # (1, seq_len + 1)
                # 2. Target Model Verifies the Candidate Tokens
                target_logits: torch.Tensor = self.target_model(candidate_sequence).logits  # (1, seq_len + 1, vocab_size)
                target_token: torch.Tensor = torch.argmax(target_logits[:, -1, :], dim=-1, keepdim=True)  # (1, 1)
            # Instead of if-else, use torch.where() to always compute both options and select dynamically
            accepted_token: torch.Tensor = torch.where(
                (draft_token == target_token), draft_token, target_token
            )  # (1, 1)
            generated_tokens = torch.cat([generated_tokens, accepted_token], dim=-1)
            # # Check EOS
            # eos_token_id = 50256  # GPT-2 EOS token
            # is_eos: torch.Tensor = (accepted_token == eos_token_id).to(torch.int)  # (1, 1), 1 if EOS else 0
            # generated_tokens = generated_tokens[:, : generated_tokens.shape[1] - int(is_eos.item())]  # Remove EOS 
            # end of version 2

        return generated_tokens
    
    
    # def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0, gamma=5, max_new_tokens=100):
    def generate(self, input_ids, pad_token_id, len_encoded_prompt, temperature=1.0, top_k=0, top_p=1.0, gamma=5, max_new_tokens=100):
        """
        Generate text using speculative decoding.

        Args:
            prompt (str): The input prompt to start generation from.
            temperature (float): The temperature for sampling. Defaults to 1.0.
            top_k (int): The number of top tokens to consider for top-k sampling. Defaults to 0 (disabled).
            top_p (float): The cumulative probability threshold for top-p sampling. Defaults to 1.0 (disabled).
            gamma (int): The number of tokens to generate speculatively in each iteration. Defaults to 5.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        for _ in range(0, max_new_tokens, gamma + 1):
            # Generate draft outputs
            # with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gamma,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_token_id,
            )
            draft_tokens = draft_outputs.sequences[:, input_ids.size(1):] #torch.Size([1, 5])
            draft_probs = torch.stack(draft_outputs.scores).softmax(-1) #torch.Size([5, 1, 50257]) for GPT2
            # Target model single forward pass
            # with torch.no_grad():
            target_outputs = self.target_model(
                torch.cat([input_ids, draft_tokens], dim=1),
                attention_mask=torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1),
                return_dict=True,
            )
            
            target_logits = target_outputs.logits[:, input_ids.size(1)-1:-1]
            target_probs = self.sample(target_logits, temperature, top_k, top_p)
            
            # Speculative sampling
            accepted_tokens = []
            for i in range(gamma):
                draft_token = draft_tokens[:, i]
                draft_prob = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                target_prob = target_probs[:, i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
                
                accept_prob = torch.min(torch.ones_like(target_prob), target_prob / draft_prob)
                if torch.rand(1, device=self.device) < accept_prob:
                    accepted_tokens.append(draft_token)
                else:
                    break
                # if draft_prob <= target_prob:
                #     # Accept deterministically if draft prob <= target_prob
                #     accepted_tokens.append(draft_token)
                # else:
                #     # Probabilistic rejection if draft prob > target_prob
                #     rejection_prob = 1 - target_prob / draft_prob
                #     if torch.rand(1, device=self.device) < rejection_prob:
                #         break
            
            num_accepted = len(accepted_tokens)
            
            if num_accepted < gamma:
                adjusted_probs = torch.clamp(target_probs[:, num_accepted] - draft_probs[num_accepted], min=0)
                adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(adjusted_probs, num_samples=1)
            else:
                next_token = torch.multinomial(target_probs[:, -1], num_samples=1)
            
            accepted_tokens.append(next_token)
            new_tokens = torch.cat([token.view(1, 1) for token in accepted_tokens], dim=1)
            
            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens)], dim=1) #update for next generation
            
            if input_ids.size(1) - len_encoded_prompt >= max_new_tokens:
                break
        
        # return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return input_ids[0]


    def target_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the target model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50.

        Returns:
            str: The generated text.
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.target_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(greedy_output[0])

    def draft_generate_greedy(self, prompt, max_new_tokens=50):
        """
        Generate text using standard greedy decoding with the draft model.

        Args:
            prompt (str): The input prompt to start generation from.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 50

        Returns:
            str: The generated text.
        """
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        greedy_output = self.draft_model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(greedy_output[0])

