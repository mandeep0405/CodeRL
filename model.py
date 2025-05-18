"""
Model setup for fine-tuning GPT-2 on code generation.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config, 
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)


class CodeGPT2:
    """Wrapper around GPT-2 for code generation."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize CodeGPT2 model.
        
        Args:
            model_name: Name of the base model
            device: Device to run on
            special_tokens: Special tokens to add to the tokenizer
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # GPT-2 tokenizer doesn't have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens
        if special_tokens:
            self.add_special_tokens(special_tokens)
        
        # Move model to device
        self.model.to(device)
    
    def add_special_tokens(self, special_tokens: List[str]) -> None:
        """
        Add special tokens to the tokenizer.
        
        Args:
            special_tokens: List of special tokens to add
        """
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate code based on the given prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated code sequences
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move each tensor in the dictionary to the device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode and return
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    def save(self, output_dir: str) -> None:
        """
        Save model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load(self, model_path: str) -> None:
        """
        Load model and tokenizer.
        
        Args:
            model_path: Path to load from
        """
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)


class CodeRLModel(CodeGPT2):
    """Extension of CodeGPT2 with RL capabilities."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize CodeRLModel.
        
        Args:
            model_name: Name of the base model
            device: Device to run on
            special_tokens: Special tokens to add to the tokenizer
        """
        super().__init__(model_name, device, special_tokens)
        
        # Keep a reference to the original model for KL divergence calculation
        self.reference_model = None
    
    def set_reference_model(self) -> None:
        """Set the current model as the reference model."""
        self.reference_model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.to(self.device)
        self.reference_model.eval()  # Set to evaluation mode
    
    def sample_batch(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Sample a batch of responses and return log probabilities.
        
        Args:
            prompts: List of prompts
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (generated_texts, log_probs, token_ids)
        """
        batch_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # Move each tensor in the dictionary to the device
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        
        # For storing log probs and token ids
        all_log_probs = []
        all_token_ids = []
        generated_texts = []
        
        for i, prompt in enumerate(prompts):
            # Tokenize single prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Move each tensor in the dictionary to the device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[1]
            
            # Generate with sampling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=input_len + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            # Get token IDs (only the generated part, not the prompt)
            token_ids = outputs.sequences[0, input_len:]
            
            # Get log probabilities for each token
            log_probs = torch.zeros(token_ids.shape[0], device=self.device)
            
            # Compute log probs token by token
            for j, (token_id, score) in enumerate(zip(token_ids, outputs.scores)):
                # Get log prob of the chosen token
                log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                log_probs[j] = log_prob
            
            all_log_probs.append(log_probs)
            all_token_ids.append(token_ids)
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # Padding log_probs and token_ids to the same length
        max_len = max(lp.shape[0] for lp in all_log_probs)
        padded_log_probs = torch.zeros(len(prompts), max_len, device=self.device)
        padded_token_ids = torch.zeros(len(prompts), max_len, dtype=torch.long, device=self.device)
        
        for i, (log_probs, token_ids) in enumerate(zip(all_log_probs, all_token_ids)):
            padded_log_probs[i, :log_probs.shape[0]] = log_probs
            padded_token_ids[i, :token_ids.shape[0]] = token_ids
        
        return generated_texts, padded_log_probs, padded_token_ids
    
    def compute_kl_divergence(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between current model and reference model.
        
        Args:
            token_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            KL divergence
        """
        if self.reference_model is None:
            raise ValueError("Reference model not set. Call set_reference_model() first.")
        
        # Get logits from current model
        with torch.no_grad():
            current_outputs = self.model(token_ids, attention_mask=attention_mask)
            current_logits = current_outputs.logits
            
            # Get logits from reference model
            ref_outputs = self.reference_model(token_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
        
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean"
        )
        
        return kl_div
    
    def update_from_ppo(
        self,
        token_ids: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_eps: float = 0.2,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01
    ) -> Dict[str, float]:
        """
        Update model using PPO algorithm.
        
        Args:
            token_ids: Token IDs
            advantages: Advantage values
            old_log_probs: Log probabilities from old policy
            clip_eps: PPO clipping parameter
            kl_coef: KL divergence coefficient
            entropy_coef: Entropy coefficient
            
        Returns:
            Dictionary with training metrics
        """
        # Compute log probs for current policy
        outputs = self.model(token_ids)
        logits = outputs.logits
        
        # Compute log probs for each token
        log_probs = F.log_softmax(logits, dim=-1)
        current_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute ratio of probabilities
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute KL divergence if reference model is set
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.reference_model is not None:
            kl_loss = self.compute_kl_divergence(token_ids)
        
        # Compute entropy bonus
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + kl_coef * kl_loss - entropy_coef * entropy
        
        # Update model
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Get training metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }
        
        return metrics