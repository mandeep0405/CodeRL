"""
Main entry point for training and evaluating code generation models.
"""

import argparse
import os
import logging
import torch
from transformers import set_seed

from code_rl_tuner.model import CodeGPT2, CodeRLModel
from code_rl_tuner.data_loaders import get_dataloaders
from code_rl_tuner.reward_model import CodeRewardModel
from code_rl_tuner.trainer import SFTTrainer, RLTrainer
from code_rl_tuner.evaluator import HumanEvalEvaluator, MBPPEvaluator


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate code generation models")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--log_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="code-rl-tuner", help="W&B project name")
    
    # Data arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum number of validation samples")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--train_type", type=str, choices=["sft", "rl", "both", "eval"], default="both", 
                      help="Training type (sft, rl, both, or eval)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for SFT")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-5, help="Learning rate for RL")
    parser.add_argument("--sft_epochs", type=int, default=3, help="Number of SFT epochs")
    parser.add_argument("--rl_iterations", type=int, default=1000, help="Number of RL iterations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # RL arguments
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per batch")
    parser.add_argument("--ppo_clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--eval_num_samples", type=int, default=1, help="Number of samples for evaluation")
    parser.add_argument("--baseline_model", type=str, default=None, help="Baseline model for comparison")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting with arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    if args.train_type in ["sft", "both"] or args.eval_only:
        model = CodeGPT2(model_name=args.model_name, device=args.device)
        logger.info(f"Initialized CodeGPT2 model with base {args.model_name}")
    else:
        model = CodeRLModel(model_name=args.model_name, device=args.device)
        logger.info(f"Initialized CodeRLModel with base {args.model_name}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_length=args.max_length
    )
    
    logger.info(f"Dataloaders created. Train size: {len(train_loader)}, Val size: {len(val_loader)}, Test size: {len(test_loader)}")
    
    # Training and evaluation paths
    if not args.eval_only:
        if args.train_type in ["sft", "both"]:
            # Train with SFT
            logger.info("Starting Supervised Fine-Tuning (SFT)")
            sft_trainer = SFTTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=args.learning_rate,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                output_dir=os.path.join(args.output_dir, "sft"),
                log_wandb=args.log_wandb,
                wandb_project=args.wandb_project
            )
            
            sft_trainer.train(num_epochs=args.sft_epochs)
            logger.info("SFT completed. Model saved at " + os.path.join(args.output_dir, "sft", "final_model"))
            
            # Load the best SFT model
            model.load(os.path.join(args.output_dir, "sft", "best_model"))
            logger.info("Loaded best SFT model for evaluation")
            
            # Evaluate the SFT model on MBPP and HumanEval
            mbpp_evaluator = MBPPEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            mbpp_results = mbpp_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name="sft_model"
            )
            
            humaneval_evaluator = HumanEvalEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            humaneval_results = humaneval_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name="sft_model"
            )
            
            logger.info("SFT model evaluation completed")
        
        if args.train_type in ["rl", "both"]:
            # If we're doing both, we already have the SFT model loaded
            if args.train_type == "rl":
                # Load pre-trained SFT model if available, otherwise use base model
                sft_path = os.path.join(args.output_dir, "sft", "best_model")
                if os.path.exists(sft_path):
                    logger.info(f"Loading pre-trained SFT model from {sft_path}")
                    model = CodeRLModel(model_name=args.model_name, device=args.device)
                    model.load(sft_path)
                else:
                    logger.info("No pre-trained SFT model found, using base model for RL")
                    model = CodeRLModel(model_name=args.model_name, device=args.device)
            
            # Initialize reward model
            reward_model = CodeRewardModel()
            
            # Train with RL
            logger.info("Starting Reinforcement Learning (RL) training")
            rl_trainer = RLTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                reward_model=reward_model,
                learning_rate=args.rl_learning_rate,
                ppo_epochs=args.ppo_epochs,
                ppo_clip=args.ppo_clip,
                kl_coef=args.kl_coef,
                entropy_coef=args.entropy_coef,
                output_dir=os.path.join(args.output_dir, "rl"),
                log_wandb=args.log_wandb,
                wandb_project=args.wandb_project
            )
            
            rl_trainer.train(num_iterations=args.rl_iterations)
            logger.info("RL training completed. Model saved at " + os.path.join(args.output_dir, "rl", "final_model"))
            
            # Load the final RL model
            model.load(os.path.join(args.output_dir, "rl", "final_model"))
            logger.info("Loaded final RL model for evaluation")
            
            # Evaluate the RL model
            mbpp_evaluator = MBPPEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            mbpp_results = mbpp_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name="rl_model"
            )
            
            humaneval_evaluator = HumanEvalEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            humaneval_results = humaneval_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name="rl_model"
            )
            
            logger.info("RL model evaluation completed")
    
    else:  # Evaluation only mode
        # Determine which models to evaluate
        models_to_evaluate = []
        
        # Add baseline model if provided
        if args.baseline_model:
            baseline_model = CodeGPT2(model_name=args.baseline_model, device=args.device)
            models_to_evaluate.append(("baseline", baseline_model))
        
        # Add SFT model if available
        sft_path = os.path.join(args.output_dir, "sft", "best_model")
        if os.path.exists(sft_path):
            sft_model = CodeGPT2(model_name=args.model_name, device=args.device)
            sft_model.load(sft_path)
            models_to_evaluate.append(("sft", sft_model))
        
        # Add RL model if available
        rl_path = os.path.join(args.output_dir, "rl", "final_model")
        if os.path.exists(rl_path):
            rl_model = CodeRLModel(model_name=args.model_name, device=args.device)
            rl_model.load(rl_path)
            models_to_evaluate.append(("rl", rl_model))
        
        if not models_to_evaluate:
            logger.error("No models found for evaluation. Please train models first or provide a baseline model.")
            return
        
        # Evaluate each model
        mbpp_results = []
        humaneval_results = []
        
        for model_name, model in models_to_evaluate:
            logger.info(f"Evaluating {model_name} model")
            
            # Evaluate on MBPP
            mbpp_evaluator = MBPPEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            result = mbpp_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name=model_name
            )
            mbpp_results.append(result)
            
            # Evaluate on HumanEval
            humaneval_evaluator = HumanEvalEvaluator(
                model=model,
                output_dir=os.path.join(args.output_dir, "eval")
            )
            
            result = humaneval_evaluator.evaluate(
                num_samples=args.eval_num_samples,
                model_name=model_name
            )
            humaneval_results.append(result)
        
        # Compare models
        if len(models_to_evaluate) > 1:
            logger.info("Comparing models")
            
            mbpp_evaluator.compare_models(mbpp_results)
            humaneval_evaluator.compare_models(humaneval_results)
    
    logger.info("All operations completed successfully")


if __name__ == "__main__":
    main()