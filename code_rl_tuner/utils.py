"""
Utility functions for the code_rl_tuner package.
"""

import os
import re
import logging
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_function_from_code(code: str) -> str:
    """
    Extract function definition from generated code.
    
    Args:
        code: Generated code
        
    Returns:
        Extracted function definition
    """
    # Try to find the function definition
    function_match = re.search(r'def\s+\w+\s*\(.*?\)(?:\s*->.*?)?\s*:.*?(?=\n\S|$)', 
                             code, re.DOTALL)
    
    if function_match:
        # Extract the function and its body with proper indentation
        function_def = function_match.group(0)
        start_idx = function_match.start()
        
        # Find the full function body by tracking indentation
        lines = code[start_idx:].split('\n')
        function_lines = [lines[0]]
        
        if len(lines) > 1:
            base_indent = len(lines[1]) - len(lines[1].lstrip())
            for line in lines[1:]:
                if line.strip() == '':
                    function_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip() != '':
                    break
                function_lines.append(line)
        
        return '\n'.join(function_lines)
    
    return code  # Return original if no function found


def check_code_safety(code: str) -> Tuple[bool, str]:
    """
    Check if code contains potentially unsafe operations.
    
    Args:
        code: Code to check
        
    Returns:
        Tuple of (is_safe, reason)
    """
    unsafe_patterns = [
        (r'import\s+os', 'imports os module'),
        (r'import\s+sys', 'imports sys module'),
        (r'import\s+subprocess', 'imports subprocess module'),
        (r'import\s+socket', 'imports socket module'),
        (r'open\s*\(', 'contains file operations'),
        (r'__import__', 'contains dynamic imports'),
        (r'eval\s*\(', 'contains eval()'),
        (r'exec\s*\(', 'contains exec()'),
        (r'os\.(system|popen|spawn|exec)', 'contains os.system or similar'),
        (r'subprocess\.(call|run|Popen)', 'contains subprocess calls'),
        (r'importlib', 'imports importlib module')
    ]
    
    for pattern, reason in unsafe_patterns:
        if re.search(pattern, code):
            return False, f"Code {reason} which may be unsafe"
    
    return True, "Code appears safe"


def plot_training_progress(
    train_losses: List[float],
    val_losses: List[float] = None,
    rewards: List[float] = None,
    pass_rates: List[float] = None,
    output_dir: str = "./plots"
) -> None:
    """
    Plot training progress.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        rewards: Rewards (for RL)
        pass_rates: Pass rates (for RL)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot rewards and pass rates (for RL)
    if rewards or pass_rates:
        plt.subplot(1, 2, 2)
        if rewards:
            plt.plot(rewards, label='Reward', color='green')
        if pass_rates:
            plt.plot(pass_rates, label='Pass Rate', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title('RL Metrics')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.close()


def plot_reward_distribution(rewards: List[float], output_dir: str = "./plots") -> None:
    """
    Plot distribution of rewards.
    
    Args:
        rewards: List of rewards
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, bins=20, kde=True)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rewards')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    plt.close()


def log_successful_example(
    prompt: str,
    code: str,
    test_cases: List[str],
    execution_results: Dict[str, Any],
    log_file: str = "successful_examples.txt"
) -> None:
    """
    Log a successful code generation example.
    
    Args:
        prompt: Problem prompt
        code: Generated code
        test_cases: Test cases
        execution_results: Execution results
        log_file: Log file path
    """
    with open(log_file, 'a') as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"CODE:\n{code}\n\n")
        f.write(f"TEST CASES:\n")
        for test in test_cases:
            f.write(f"- {test}\n")
        f.write("\n")
        f.write(f"EXECUTION RESULTS:\n")
        f.write(f"Success: {execution_results['success']}\n")
        if 'pass_rate' in execution_results:
            f.write(f"Pass Rate: {execution_results['pass_rate']}\n")
        f.write(f"{'=' * 80}\n\n")


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in a human-readable format.
    
    Args:
        seconds: Execution time in seconds
        
    Returns:
        Formatted execution time
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def truncate_code_for_display(code: str, max_lines: int = 30) -> str:
    """
    Truncate code for display purposes.
    
    Args:
        code: Code to truncate
        max_lines: Maximum number of lines
        
    Returns:
        Truncated code
    """
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code
    
    return '\n'.join(lines[:max_lines]) + f"\n... [truncated, {len(lines) - max_lines} more lines]"


class CodeMetrics:
    """Metrics for code quality and complexity."""
    
    @staticmethod
    def complexity(code: str) -> int:
        """
        Calculate cyclomatic complexity.
        
        Args:
            code: Code to analyze
            
        Returns:
            Cyclomatic complexity
        """
        # Simple approximation of cyclomatic complexity
        # More accurate implementation would use the ast module
        complexity = 1  # Base complexity
        
        # Count control flow statements
        control_patterns = [
            r'\sif\s+', 
            r'\selse\s*:', 
            r'\selif\s+',
            r'\sfor\s+', 
            r'\swhile\s+', 
            r'\sexcept\s*',
            r'\sand\s+', 
            r'\sor\s+'
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    @staticmethod
    def count_lines(code: str) -> int:
        """
        Count non-empty lines of code.
        
        Args:
            code: Code to analyze
            
        Returns:
            Number of non-empty lines
        """
        return len([line for line in code.split('\n') if line.strip()])
    
    @staticmethod
    def has_docstring(code: str) -> bool:
        """
        Check if code has a docstring.
        
        Args:
            code: Code to analyze
            
        Returns:
            Whether code has a docstring
        """
        # Check for triple quotes after function definition
        docstring_pattern = r'def\s+\w+\s*\(.*?\)(?:\s*->.*?)?\s*:\s*(?:\'\'\'|""")'
        return bool(re.search(docstring_pattern, code, re.DOTALL))
    
    @staticmethod
    def has_type_hints(code: str) -> bool:
        """
        Check if code has type hints.
        
        Args:
            code: Code to analyze
            
        Returns:
            Whether code has type hints
        """
        # Check for parameter type annotations or return type annotations
        type_pattern = r'def\s+\w+\s*\(.*?:.*?\)|def\s+\w+\s*\(.*?\)\s*->.*?:'
        return bool(re.search(type_pattern, code))