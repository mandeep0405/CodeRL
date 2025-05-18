"""
Training routines for fine-tuning models on code generation.
"""

import os
import time
import logging
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import wandb

from code_rl_tuner.model import CodeGPT2, CodeRLModel
from code_rl_tuner.reward_model import CodeRewardModel
from code_rl_tuner.environment import CodeExecutionEnvironment


class SFTTrainer:
    """Supervised Fine-Tuning (SFT) Trainer for code generation models."""
    
    def __init__(
        self,
        model: CodeGPT2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "./output",
        log_wandb: bool = False,
        wandb_project: str = "code-rl-tuner"
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            gradient_accumulation_steps: Number of steps for gradient accumulation
            max_grad_norm: Maximum gradient norm
            output_dir: Output directory
            log_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize W&B
        if log_wandb:
            wandb.init(project=wandb_project, config={
                "model_name": model.model_name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm
            })
    
    def train(self, num_epochs, save_steps=None, eval_steps=None, logging_steps=10):
        """
        Train the model with enhanced progress visualization.
        """
        import time
        from tqdm.auto import tqdm  # Use tqdm.auto for both notebook and terminal support
        
        # Set up optimizer
        if not hasattr(self, 'optimizer'):
            self.optimizer = AdamW(self.model.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        global_step = 0
        
        # Create epoch progress bar
        epoch_pbar = tqdm(total=num_epochs, desc="Training Epochs", position=0)
        
        # Log training start
        total_start_time = time.time()
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.model.model.train()
            epoch_loss = 0.0
            
            # Create batch progress bar
            batch_pbar = tqdm(
                total=len(self.train_loader), 
                desc=f"Epoch {epoch+1}/{num_epochs}", 
                position=1, 
                leave=False
            )
            
            epoch_start_time = time.time()
            batch_start_time = time.time()
            samples_processed = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Get tokenized inputs
                tokenized_inputs = batch["tokenized_full"]
                input_ids = tokenized_inputs["input_ids"].to(self.model.device)
                attention_mask = tokenized_inputs["attention_mask"].to(self.model.device)
                
                # Track batch size for metrics
                batch_size = input_ids.size(0)
                samples_processed += batch_size
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # Use input_ids as labels for causal LM training
                )
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                
                # Update parameters
                self.optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                global_step += 1
                
                # Update batch progress bar
                batch_pbar.update(1)
                
                # Calculate processing speed
                if (batch_idx + 1) % logging_steps == 0 or batch_idx == len(self.train_loader) - 1:
                    batch_time = time.time() - batch_start_time
                    samples_per_sec = samples_processed / batch_time
                    avg_loss = epoch_loss / (batch_idx + 1)
                    
                    # Update progress bar description
                    batch_pbar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'samples/sec': f"{samples_per_sec:.2f}"
                    })
                    
                    # Log to console
                    logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                            f"Loss: {avg_loss:.4f}, Speed: {samples_per_sec:.2f} samples/sec")
                    
                    if self.log_wandb:
                        wandb.log({
                            "train/loss": avg_loss, 
                            "train/samples_per_sec": samples_per_sec,
                            "epoch": epoch + 1, 
                            "step": global_step
                        })
                    
                    # Reset for next logging interval
                    batch_start_time = time.time()
                    samples_processed = 0
                
                # Save model
                if save_steps and global_step % save_steps == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                    self.model.save(save_path)
                    logging.info(f"Model saved at {save_path}")
                
                # Evaluate model
                if eval_steps and global_step % eval_steps == 0:
                    batch_pbar.set_description(f"Evaluating...")
                    val_loss = self.evaluate()
                    batch_pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(self.output_dir, "best_model")
                        self.model.save(save_path)
                        logging.info(f"Best model saved at {save_path} with val loss {val_loss:.4f}")
                    
                    self.model.model.train()  # Set back to train mode
            
            # Close batch progress bar
            batch_pbar.close()
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            # Evaluate after each epoch
            logging.info(f"Evaluating after epoch {epoch+1}...")
            val_loss = self.evaluate()
            
            # Log epoch results
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s. "
                    f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.output_dir, "best_model")
                self.model.save(save_path)
                logging.info(f"Best model saved at {save_path} with val loss {val_loss:.4f}")
            
            # Save final model for this epoch
            save_path = os.path.join(self.output_dir, f"epoch-{epoch+1}")
            self.model.save(save_path)
            
            # Update epoch progress bar
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'train_loss': f"{avg_epoch_loss:.4f}",
                'val_loss': f"{val_loss:.4f}"
            })
        
        # Close epoch progress bar
        epoch_pbar.close()
        
        # Save final model
        final_path = os.path.join(self.output_dir, "final_model")
        self.model.save(final_path)
        
        # Log training summary
        total_time = time.time() - total_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
        logging.info(f"Final model saved at {final_path}")
        
        return global_step, best_val_loss

    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Get tokenized inputs - MODIFIED TO HANDLE DICTIONARY STRUCTURE
                tokenized_inputs = batch["tokenized_full"] if "tokenized_full" in batch else batch["tokenized_prompt"]
                input_ids = tokenized_inputs["input_ids"].to(self.model.device)
                attention_mask = tokenized_inputs["attention_mask"].to(self.model.device)
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # Use input_ids as labels for causal LM training
                )
                loss = outputs.loss
                
                # Track loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        logging.info(f"Validation Loss: {avg_loss:.4f}")
        
        if self.log_wandb:
            wandb.log({"val/loss": avg_loss})
        
        return avg_loss


class RLTrainer:
    """Reinforcement Learning Trainer for code generation models."""
    
    def __init__(
        self,
        model: CodeRLModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        reward_model: CodeRewardModel,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        output_dir: str = "./output_rl",
        log_wandb: bool = False,
        wandb_project: str = "code-rl-tuner"
    ):
        """
        Initialize RL trainer.
        
        Args:
            model: Model to fine-tune with RL
            train_loader: Training data loader
            val_loader: Validation data loader
            reward_model: Reward model for RL
            learning_rate: Learning rate
            ppo_epochs: Number of PPO epochs per batch
            ppo_clip: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            kl_coef: KL divergence coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            output_dir: Output directory
            log_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.reward_model = reward_model
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.value_loss_coef = value_loss_coef
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate
        )
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the environment
        self.environment = CodeExecutionEnvironment()
        
        # Initialize W&B
        if log_wandb:
            wandb.init(project=wandb_project, config={
                "model_name": model.model_name,
                "learning_rate": learning_rate,
                "ppo_epochs": ppo_epochs,
                "ppo_clip": ppo_clip,
                "value_loss_coef": value_loss_coef,
                "kl_coef": kl_coef,
                "entropy_coef": entropy_coef,
                "max_grad_norm": max_grad_norm
            })
    
    def train(
        self,
        num_iterations: int = 1000,
        samples_per_iteration: int = 16,
        save_steps: int = 100,
        eval_steps: int = 50,
        logging_steps: int = 10
    ) -> None:
        """
        Train the model using RL.
        
        Args:
            num_iterations: Number of iterations
            samples_per_iteration: Number of samples per iteration
            save_steps: Steps between saving model
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set reference model for KL divergence calculation
        self.model.set_reference_model()
        
        for iteration in range(num_iterations):
            self.logger.info(f"Starting iteration {iteration+1}/{num_iterations}")
            
            # Sample batch of prompts from loader
            prompts = []
            test_cases_list = []
            task_ids = []
            
            # Get a batch of samples
            batch = next(iter(self.train_loader))
            for sample in batch:
                prompts.append(sample["prompt"])
                test_cases_list.append(sample["test_cases"])
                task_ids.append(sample["task_id"])
            
            # Generate responses and get log probs
            self.model.model.eval()  # Set to eval for generation
            responses, log_probs, token_ids = self.model.sample_batch(
                prompts,
                max_length=256,
                temperature=0.8,
                top_p=0.9
            )
            
            # Extract code from responses
            codes = []
            for response in responses:
                # Find where the generated code begins after the prompt
                try:
                    # Assuming code starts with "def" or similar
                    if "def " in response:
                        code_start = response.index("def ")
                        code = response[code_start:]
                    else:
                        code = response
                except ValueError:
                    code = response
                
                codes.append(code)
            
            # Compute rewards
            rewards = self.reward_model.batch_compute_rewards(
                codes=codes,
                test_cases_list=test_cases_list,
                is_mbpp_list=[True] * len(codes)
            )
            
            # Convert rewards to tensor
            rewards_tensor = torch.tensor(rewards, device=self.model.device)
            
            # Log rewards
            if self.log_wandb:
                wandb.log({
                    "rl/mean_reward": rewards_tensor.mean().item(),
                    "rl/max_reward": rewards_tensor.max().item(),
                    "rl/min_reward": rewards_tensor.min().item()
                }, step=iteration)
            
            if iteration % logging_steps == 0:
                self.logger.info(f"Iteration {iteration+1}: mean_reward = {rewards_tensor.mean().item()}")
                
                # Log an example
                example_idx = rewards_tensor.argmax().item()
                self.logger.info(f"Best example (reward={rewards[example_idx]}):")
                self.logger.info(f"Prompt: {prompts[example_idx]}")
                self.logger.info(f"Generated code: {codes[example_idx]}")
            
            # PPO training loop
            self.model.model.train()
            
            # Compute advantages (using rewards as returns for simplicity)
            # In a more complete implementation, we would use a value function to estimate returns
            advantages = rewards_tensor
            
            # Run PPO for multiple epochs on the same batch
            for ppo_epoch in range(self.ppo_epochs):
                # Update model with PPO
                metrics = self.model.update_from_ppo(
                    token_ids=token_ids,
                    advantages=advantages,
                    old_log_probs=log_probs,
                    clip_eps=self.ppo_clip,
                    kl_coef=self.kl_coef,
                    entropy_coef=self.entropy_coef
                )
                
                # Apply gradient
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Log PPO metrics
                if self.log_wandb:
                    wandb.log({
                        f"ppo/policy_loss_epoch_{ppo_epoch}": metrics["policy_loss"],
                        f"ppo/kl_loss_epoch_{ppo_epoch}": metrics["kl_loss"],
                        f"ppo/entropy_epoch_{ppo_epoch}": metrics["entropy"],
                        f"ppo/total_loss_epoch_{ppo_epoch}": metrics["total_loss"]
                    }, step=iteration)
            
            # Evaluation
            if iteration % eval_steps == 0:
                eval_metrics = self.evaluate()
                self.logger.info(f"Iteration {iteration+1}: eval_reward = {eval_metrics['mean_reward']}")
                if self.log_wandb:
                    wandb.log({
                        "eval/mean_reward": eval_metrics["mean_reward"],
                        "eval/pass_rate": eval_metrics["pass_rate"]
                    }, step=iteration)
            
            # Save checkpoint
            if iteration % save_steps == 0:
                self.model.save(os.path.join(self.output_dir, f"checkpoint-{iteration}"))
                self.logger.info(f"Checkpoint saved at iteration {iteration+1}")
        
        # Save final model
        self.model.save(os.path.join(self.output_dir, "final_model"))
        self.logger.info("RL Training completed")
    
    def evaluate(self, num_samples: int = 50) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.model.eval()
        rewards = []
        pass_rates = []
        
        # Sample from validation set
        val_samples = []
        for batch in self.val_loader:
            val_samples.extend(batch)
            if len(val_samples) >= num_samples:
                break
        
        val_samples = val_samples[:num_samples]
        
        for sample in tqdm(val_samples, desc="Evaluation"):
            prompt = sample["prompt"]
            test_cases = sample["test_cases"]
            
            # Generate code
            generated = self.model.generate(
                prompt,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )[0]
            
            # Extract code
            try:
                if "def " in generated:
                    code_start = generated.index("def ")
                    code = generated[code_start:]
                else:
                    code = generated
            except ValueError:
                code = generated
            
            # Compute reward
            reward = self.reward_model.compute_reward(
                code=code,
                test_cases=test_cases,
                is_mbpp=True
            )
            rewards.append(reward)
            
            # Run test cases to get pass rate
            execution_results = self.environment.execute_mbpp_test_cases(code, test_cases)
            pass_rates.append(execution_results["pass_rate"])
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        mean_pass_rate = np.mean(pass_rates)
        
        # Set back to training mode
        self.model.model.train()
        
        return {
            "mean_reward": mean_reward,
            "pass_rate": mean_pass_rate,
            "rewards": rewards,
            "pass_rates": pass_rates
        }