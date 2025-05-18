"""
Evaluator for code generation models on HumanEval.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

from code_rl_tuner.model import CodeGPT2, CodeRLModel
from code_rl_tuner.environment import CodeExecutionEnvironment
from code_rl_tuner.data_loaders import HumanEvalDataset


class HumanEvalEvaluator:
    """Evaluator for the HumanEval benchmark."""
    
    def __init__(
        self,
        model: Union[CodeGPT2, CodeRLModel],
        output_dir: str = "./evaluation",
        timeout: int = 10
    ):
        """
        Initialize HumanEval evaluator.
        
        Args:
            model: Model to evaluate
            output_dir: Output directory for evaluation results
            timeout: Timeout for code execution
        """
        self.model = model
        self.output_dir = output_dir
        self.environment = CodeExecutionEnvironment(timeout=timeout)
        self.dataset = HumanEvalDataset(tokenizer=model.tokenizer)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        num_samples: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_length: int = 512,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on HumanEval.
        
        Args:
            num_samples: Number of samples to generate per problem
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            max_length: Maximum length of generated code
            model_name: Name for the model in results
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating model {model_name} on HumanEval")
        results = []
        idx = 0
        for idx, sample in enumerate(tqdm(self.dataset, desc="Evaluating HumanEval")):
            if idx == 10:
                break
            
            prompt = sample["prompt"]
            test_cases = sample["test_cases"]
            entry_point = sample["entry_point"]
            task_id = sample["task_id"]
            
            # Generate multiple samples
            generated_codes = []
            for _ in range(num_samples):
                generated = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1
                )[0]
                
                # Extract the generated code (everything after the prompt)
                prompt_len = len(prompt)
                generated_code = generated[prompt_len:]
                
                # Add to list
                generated_codes.append(generated_code)
            

            
            # Evaluate each generated sample
            sample_results = []
            for i, code in enumerate(generated_codes):
                # Combine prompt and generated code
                full_code = prompt + code
                
                # Execute the code against the test cases
                execution_result = self.environment.execute_humaneval_test(
                    code=full_code,
                    test_code=test_cases,
                    entry_point=entry_point
                )
                
                # Record result
                sample_result = {
                    "sample_id": i,
                    "success": execution_result["success"],
                    "compile_error": execution_result.get("compile_error"),
                    "runtime_error": execution_result.get("runtime_error"),
                    "code": code
                }
                sample_results.append(sample_result)
            
            # Compute pass rate for this problem
            pass_rate = sum(1 for r in sample_results if r["success"]) / num_samples
            
            # Add to results
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "pass_rate": pass_rate,
                "samples": sample_results
            }
            results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(self.dataset)} problems")
        
        # Calculate overall metrics
        problem_success = [r["pass_rate"] > 0 for r in results]  # At least one sample passes
        pass_at_k = {
            "pass@1": np.mean([r["pass_rate"] for r in results]),
            "problems_solved": sum(problem_success),
            "total_problems": len(results)
        }
        
        # If we have multiple samples, calculate pass@k metrics
        if num_samples > 1:
            for k in [1, 5, 10, 20]:
                if k <= num_samples:
                    # pass@k - probability of solving if we sample k times
                    # Using formula from the HumanEval paper
                    n = num_samples
                    pass_at_k[f"pass@{k}"] = np.mean([1 - (1 - r["pass_rate"]) ** min(k, n) for r in results])
        
        # Complete results
        evaluation_results = {
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "num_samples": num_samples,
            "metrics": pass_at_k,
            "detailed_results": results
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, f"{model_name}_humaneval_results.json")
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to {results_path}")
        self.logger.info(f"Pass@1: {pass_at_k['pass@1']}")
        self.logger.info(f"Problems solved: {pass_at_k['problems_solved']}/{pass_at_k['total_problems']}")
        
        return evaluation_results
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> None:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation results from different models
        """
        # Extract model names and pass@1 scores
        model_names = [r["model_name"] for r in model_results]
        pass_at_1 = [r["metrics"]["pass@1"] for r in model_results]
        problems_solved = [r["metrics"]["problems_solved"] for r in model_results]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot pass@1
        ax1.bar(model_names, pass_at_1)
        ax1.set_title("HumanEval Pass@1")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Pass@1")
        ax1.grid(axis="y", alpha=0.3)
        
        # Plot problems solved
        ax2.bar(model_names, problems_solved)
        ax2.set_title("HumanEval Problems Solved")
        ax2.set_ylim(0, model_results[0]["metrics"]["total_problems"])
        ax2.set_ylabel("Problems Solved")
        ax2.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(pass_at_1):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha="center")
        
        for i, v in enumerate(problems_solved):
            ax2.text(i, v + 1, str(v), ha="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        plt.close()
        
        self.logger.info(f"Model comparison plot saved to {os.path.join(self.output_dir, 'model_comparison.png')}")
        
        # Create detailed comparison
        comparison = {
            "models": [],
            "total_problems": model_results[0]["metrics"]["total_problems"]
        }
        
        for result in model_results:
            model_data = {
                "name": result["model_name"],
                "pass@1": result["metrics"]["pass@1"],
                "problems_solved": result["metrics"]["problems_solved"]
            }
            
            # Add pass@k metrics if available
            for k in [5, 10, 20]:
                if f"pass@{k}" in result["metrics"]:
                    model_data[f"pass@{k}"] = result["metrics"][f"pass@{k}"]
            
            comparison["models"].append(model_data)
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"Detailed comparison saved to {comparison_path}")


class MBPPEvaluator:
    """Evaluator for the MBPP benchmark."""
    
    def __init__(
        self,
        model: Union[CodeGPT2, CodeRLModel],
        output_dir: str = "./evaluation",
        timeout: int = 10
    ):
        """
        Initialize MBPP evaluator.
        
        Args:
            model: Model to evaluate
            output_dir: Output directory for evaluation results
            timeout: Timeout for code execution
        """
        self.model = model
        self.output_dir = output_dir
        self.environment = CodeExecutionEnvironment(timeout=timeout)
        
        # Load MBPP test dataset
        from code_rl_tuner.data_loaders import MBPPDataset
        self.dataset = MBPPDataset(split="validation", tokenizer=model.tokenizer)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        num_samples: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_length: int = 512,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on MBPP.
        
        Args:
            num_samples: Number of samples to generate per problem
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            max_length: Maximum length of generated code
            model_name: Name for the model in results
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating model {model_name} on MBPP")
        results = []
        
        for idx, sample in enumerate(tqdm(self.dataset, desc="Evaluating MBPP")):
            prompt = sample["prompt"]
            test_cases = sample["test_cases"]
            task_id = sample["task_id"]
            
            # Generate multiple samples
            generated_codes = []
            for _ in range(num_samples):
                generated = self.model.generate(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1
                )[0]
                
                # Extract the generated code
                try:
                    if "def " in generated:
                        code_start = generated.index("def ")
                        generated_code = generated[code_start:]
                    else:
                        generated_code = generated
                except ValueError:
                    generated_code = generated
                
                # Add to list
                generated_codes.append(generated_code)
            
            # Evaluate each generated sample
            sample_results = []
            for i, code in enumerate(generated_codes):
                # Execute the code against the test cases
                execution_result = self.environment.execute_mbpp_test_cases(
                    code=code,
                    test_cases=test_cases
                )
                
                # Record result
                sample_result = {
                    "sample_id": i,
                    "success": execution_result["success"],
                    "pass_rate": execution_result["pass_rate"],
                    "compile_error": execution_result.get("compile_error"),
                    "code": code
                }
                sample_results.append(sample_result)
            
            # Compute pass rate for this problem
            pass_rate = sum(1 for r in sample_results if r["success"]) / num_samples
            
            # Add to results
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "pass_rate": pass_rate,
                "samples": sample_results
            }
            results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(self.dataset)} problems")
        
        # Calculate overall metrics
        problem_success = [r["pass_rate"] > 0 for r in results]  # At least one sample passes
        pass_at_k = {
            "pass@1": np.mean([r["pass_rate"] for r in results]),
            "problems_solved": sum(problem_success),
            "total_problems": len(results)
        }
        
        # If we have multiple samples, calculate pass@k metrics
        if num_samples > 1:
            for k in [1, 5, 10, 20]:
                if k <= num_samples:
                    # pass@k - probability of solving if we sample k times
                    n = num_samples
                    pass_at_k[f"pass@{k}"] = np.mean([1 - (1 - r["pass_rate"]) ** min(k, n) for r in results])
        
        # Complete results
        evaluation_results = {
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "num_samples": num_samples,
            "metrics": pass_at_k,
            "detailed_results": results
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, f"{model_name}_mbpp_results.json")
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to {results_path}")
        self.logger.info(f"Pass@1: {pass_at_k['pass@1']}")
        self.logger.info(f"Problems solved: {pass_at_k['problems_solved']}/{pass_at_k['total_problems']}")
        
        return evaluation_results