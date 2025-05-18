import sys
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader

# If you want to import from your existing code_rl_tuner package
# sys.path.append("/path/to/your/code_rl_tuner/directory")
# from code_rl_tuner.data_loaders import HumanEvalDataset

# Modified MBPPDataset class that uses the direct JSON file
class MBPPDataset(Dataset):
    """Dataset wrapper for MBPP (Mostly Basic Python Problems)."""
    
    def __init__(
        self, 
        split: str = "train", 
        max_samples: Optional[int] = None,
        tokenizer = None,
        max_length: int = 512,
    ):
        """
        Initialize MBPP dataset.
        
        Args:
            split: Dataset split ("train" or "validation" or "test")
            max_samples: Maximum number of samples to load
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        # Load dataset from GitHub raw file
        self.split = split
        self.dataset = load_dataset("json", data_files="https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl")
        
        if max_samples and max_samples < len(self.dataset):
            indices = random.sample(range(len(self.dataset)), max_samples)
            self.dataset = self.dataset.select(indices)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_dataset()
    
    def process_dataset(self):
        """Process and format the dataset."""
        self.processed_data = []
        print(f"Processing {self.split} dataset with {len(self.dataset[self.split])} samples.")
        for item in self.dataset[self.split]:
            # Format prompt with task description 
            prompt = f"# {item['text']}\ndef "
            
            # Format solution
            solution = item['code']
            if not solution.startswith("def"):
                # Extract function signature from the solution
                try:
                    fn_start = solution.index("def ")
                    fn_end = solution.index(":", fn_start)
                    fn_signature = solution[fn_start:fn_end+1]
                    
                    # Construct the full prompt
                    full_prompt = f"{prompt}\n{solution}"
                except ValueError:
                    # Fallback if parsing fails
                    full_prompt = f"{prompt}\n{solution}"
            else:
                full_prompt = f"# {item['text']}\n{solution}"
            
            # Store test cases for evaluation
            test_cases = item['test_list'] if 'test_list' in item else []
            
            entry = {
                "prompt": prompt,
                "solution": solution,
                "full_prompt": full_prompt,
                "test_cases": test_cases,
                "task_id": item.get('task_id', str(item.get('id', '')))
            }
            
            if self.tokenizer:
                entry["tokenized_prompt"] = self.tokenizer(
                    prompt, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                entry["tokenized_full"] = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            self.processed_data.append(entry)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]


class HumanEvalDataset(Dataset):
    """Dataset wrapper for HumanEval."""

    def __init__(
        self,
        tokenizer = None,
        max_length: int = 512,
    ):
        """
        Initialize HumanEval dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.dataset = load_dataset("openai_humaneval", split="test")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_dataset()
    
    def process_dataset(self):
        """Process and format the dataset."""
        self.processed_data = []
        
        for item in self.dataset:
            # Extract prompt and canonical solution
            prompt = item['prompt']
            canonical_solution = item['canonical_solution']
            entry_point = item['entry_point']
            test_cases = item['test']  # Contains test code
            
            entry = {
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "entry_point": entry_point,
                "test_cases": test_cases,
                "task_id": item['task_id']
            }
            
            if self.tokenizer:
                entry["tokenized_prompt"] = self.tokenizer(
                    prompt, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            self.processed_data.append(entry)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

