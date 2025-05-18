"""
Reward model for reinforcement learning-based fine-tuning of code generation models.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from code_rl_tuner.environment import CodeExecutionEnvironment


class CodeRewardModel:
    """Model for computing rewards for generated code."""
    
    def __init__(self, execution_timeout: int = 5):
        """
        Initialize the reward model.
        
        Args:
            execution_timeout: Timeout for code execution
        """
        self.environment = CodeExecutionEnvironment(timeout=execution_timeout)
    
    def compute_reward(self, 
                       code: str, 
                       test_cases: List[str], 
                       is_mbpp: bool = True,
                       entry_point: Optional[str] = None) -> float:
        """
        Compute reward for generated code.
        
        Args:
            code: Generated code
            test_cases: Test cases (MBPP format or HumanEval test code)
            is_mbpp: Whether the test cases are in MBPP format
            entry_point: Entry point function name (for HumanEval)
            
        Returns:
            Reward value
        """
        if is_mbpp:
            execution_results = self.environment.execute_mbpp_test_cases(code, test_cases)
        else:
            assert entry_point is not None, "Entry point required for HumanEval"
            execution_results = self.environment.execute_humaneval_test(code, test_cases, entry_point)
        
        return self._calculate_reward(execution_results, is_mbpp)
    
    def _calculate_reward(self, execution_results: Dict[str, Any], is_mbpp: bool) -> float:
        """
        Calculate reward based on execution results.
        
        Args:
            execution_results: Results from code execution
            is_mbpp: Whether the results are from MBPP format
            
        Returns:
            Reward value
        """
        # Base reward structure:
        # - Compilation success: +0.25
        # - Each passing test: +0.5 / num_tests
        # - All tests passing: +0.25 bonus
        # - Compilation error: -0.5
        # - Runtime error: -0.25
        
        if is_mbpp:
            # Get results for MBPP format
            if execution_results.get("compile_error"):
                # Compilation error
                return -0.5
            
            # Start with compilation success reward
            reward = 0.25
            
            # Add reward for passing tests
            test_results = execution_results.get("test_results", [])
            num_tests = max(1, len(test_results))
            
            for result in test_results:
                if result["success"] and not result.get("runtime_error"):
                    reward += 0.5 / num_tests
            
            # Add bonus for all tests passing
            if execution_results["success"]:
                reward += 0.25
            
            return min(1.0, reward)  # Cap reward at 1.0
        else:
            # Get results for HumanEval format
            if execution_results.get("compile_error"):
                # Compilation error
                return -0.5
            
            if execution_results.get("runtime_error"):
                # Runtime error
                return -0.25
            
            # Success means all tests passed
            if execution_results["success"]:
                return 1.0
            
            # Some tests may have passed
            return 0.25  # Partial credit
    
    def batch_compute_rewards(self, 
                             codes: List[str], 
                             test_cases_list: List[List[str]],
                             is_mbpp_list: Optional[List[bool]] = None,
                             entry_points: Optional[List[str]] = None) -> List[float]:
        """
        Compute rewards for a batch of generated codes.
        
        Args:
            codes: List of generated codes
            test_cases_list: List of test cases for each code
            is_mbpp_list: Whether each test case is in MBPP format
            entry_points: List of entry point function names (for HumanEval)
            
        Returns:
            List of reward values
        """
        if is_mbpp_list is None:
            is_mbpp_list = [True] * len(codes)
        
        rewards = []
        for i, code in enumerate(codes):
            is_mbpp = is_mbpp_list[i]
            test_cases = test_cases_list[i]
            entry_point = None if entry_points is None else entry_points[i]
            
            reward = self.compute_reward(
                code,
                test_cases,
                is_mbpp=is_mbpp,
                entry_point=entry_point
            )
            rewards.append(reward)
        
        return rewards


class LearnedRewardModel(nn.Module):
    """Learned reward model for code quality beyond just functional correctness."""
    
    def __init__(self, base_model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize learned reward model.
        
        Args:
            base_model: Base language model
            tokenizer: Tokenizer for the model
            device: Device to run the model on
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        
        # Use the base model's encoder
        self.encoder = base_model.get_encoder()
        
        # Add reward head
        hidden_size = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.execution_reward_model = CodeRewardModel()
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass to compute reward score.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Reward score
        """
        # Get the encoder outputs
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use the [CLS] token or mean of all tokens
        pooled = hidden_states[:, 0, :]  # [CLS] token
        
        # Compute reward score
        reward = self.reward_head(pooled)
        return reward
    
    def compute_reward(self, code: str, test_cases: List[str], **kwargs) -> Tuple[float, float]:
        """
        Compute combined reward for generated code.
        
        Args:
            code: Generated code
            test_cases: Test cases
            **kwargs: Additional arguments for execution reward
            
        Returns:
            Tuple of (combined_reward, execution_reward)
        """
        # Get execution reward
        execution_reward = self.execution_reward_model.compute_reward(code, test_cases, **kwargs)
        
        # Get style/quality reward (syntax, readability, etc.)
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            quality_score = self.forward(**inputs).item()
        
        # Scale quality score to [0, 0.5]
        quality_score = 0.5 * torch.sigmoid(torch.tensor(quality_score)).item()
        
        # Combined reward: execution (0 to 1) + quality (0 to 0.5)
        combined_reward = execution_reward + quality_score
        
        return combined_reward, execution_reward