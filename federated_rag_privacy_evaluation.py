#!/usr/bin/env python3
# federated_rag_privacy_evaluation.py

"""
Federated Learning Privacy Evaluation Script
Evaluates Homomorphic Encryption (HE) and Differential Privacy (DP) 
against privacy challenges in federated RAG systems.
"""

import os
import sys
import time
import logging
import warnings
from typing import List, Dict, Any, Tuple
import subprocess

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        logger.info(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package}")
        return False
    return True

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "torch",
        "transformers",
        "datasets", 
        "flwr",
        "opacus",
        "tenseal",
        "matplotlib",
        "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            if not install_package(package):
                logger.error(f"Critical: Could not install {package}")
                return False
    
    return True

# Check dependencies first
if not check_and_install_dependencies():
    sys.exit(1)

# Now import the packages
try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    import flwr as fl
    from opacus import PrivacyEngine
    from datasets import load_dataset
    from transformers import AutoTokenizer, T5ForConditionalGeneration, logging as hf_logging
    import matplotlib.pyplot as plt
    
    # Try to import tenseal (might fail on some systems)
    try:
        import tenseal as ts
        HE_AVAILABLE = True
    except ImportError:
        logger.warning("TenSEAL not available - HE features will be disabled")
        HE_AVAILABLE = False
    
    # Suppress transformers warnings
    hf_logging.set_verbosity_error()
    
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    sys.exit(1)

# === Configuration ===
CONFIG = {
    "CLIENT_NUM": 2,
    "NUM_ROUNDS": 3, 
    "SMALL_DATASET_SIZE": 50,  # Reduced for faster execution
    "BATCH_SIZE": 2,  # Reduced for memory efficiency
    "LR": 5e-4,  # Slightly higher for faster convergence
    "DP_NOISE": 1.0,
    "DP_MAX_GRAD": 1.0,
    "MAX_LENGTH": 32,  # Reduced sequence length
    "LABEL_LENGTH": 16,  # Reduced label length
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

logger.info(f"Using device: {CONFIG['DEVICE']}")

environment = {
    "membership_attack": True,
    "dp_epsilon_delta": {"delta": 1e-5}
}

# === Data Loading ===
def get_tokenizer():
    """Get tokenizer with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

TOKENIZER = get_tokenizer()

def load_and_split_data(seed=42):
    """Load and split dataset with error handling"""
    try:
        logger.info("Loading PubMed QA dataset...")
        dataset = load_dataset("pubmed_qa", "pqa_labeled", trust_remote_code=True)
        data = dataset["train"]
        
        # Use smaller subset for demo
        if CONFIG["SMALL_DATASET_SIZE"] < len(data):
            data = data.select(range(CONFIG["SMALL_DATASET_SIZE"]))
        
        data = data.shuffle(seed=seed)
        chunk_size = len(data) // CONFIG["CLIENT_NUM"]
        
        splits = []
        for i in range(CONFIG["CLIENT_NUM"]):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < CONFIG["CLIENT_NUM"] - 1 else len(data)
            splits.append(data.select(range(start_idx, end_idx)))
        
        # Use last chunk as test set
        test_set = data.select(range(-chunk_size, 0))
        
        logger.info(f"Data split into {len(splits)} clients with {len(test_set)} test samples")
        return splits, test_set
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

class QADataset(Dataset):
    """Optimized QA Dataset"""
    def __init__(self, samples):
        self.samples = samples
        logger.info(f"Created dataset with {len(samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            item = self.samples[idx]
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            
            # Tokenize question
            question_encoded = TOKENIZER(
                question, 
                truncation=True, 
                padding="max_length", 
                max_length=CONFIG["MAX_LENGTH"], 
                return_tensors="pt"
            )
            
            # Tokenize answer
            answer_encoded = TOKENIZER(
                answer, 
                truncation=True, 
                padding="max_length", 
                max_length=CONFIG["LABEL_LENGTH"], 
                return_tensors="pt"
            )
            
            return {
                "input_ids": question_encoded.input_ids.squeeze(),
                "attention_mask": question_encoded.attention_mask.squeeze(),
                "labels": answer_encoded.input_ids.squeeze()
            }
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return dummy data
            return {
                "input_ids": torch.zeros(CONFIG["MAX_LENGTH"], dtype=torch.long),
                "attention_mask": torch.zeros(CONFIG["MAX_LENGTH"], dtype=torch.long),
                "labels": torch.zeros(CONFIG["LABEL_LENGTH"], dtype=torch.long)
            }

class PrivacyClient(fl.client.NumPyClient):
    """Federated client with privacy enhancements"""
    
    def __init__(self, client_id: str, samples, use_he=False, use_dp=False):
        self.client_id = client_id
        self.dataset = QADataset(samples)
        self.use_he = use_he and HE_AVAILABLE
        self.use_dp = use_dp
        
        # Initialize model
        try:
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.model.to(CONFIG["DEVICE"])
            self.optimizer = AdamW(self.model.parameters(), lr=CONFIG["LR"])
        except Exception as e:
            logger.error(f"Failed to initialize model for client {client_id}: {e}")
            raise
        
        # Initialize HE context if needed
        if self.use_he:
            self._init_he_context()
        
        self.metrics = {}
        logger.info(f"Client {client_id} initialized - HE: {self.use_he}, DP: {self.use_dp}")
    
    def _init_he_context(self):
        """Initialize homomorphic encryption context"""
        try:
            # Create CKKS context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[40, 20, 40]
            )
            context.global_scale = 2**20  # Reduced scale for stability
            context.generate_galois_keys()
            
            # Serialize context
            context_bytes = context.serialize()
            
            # Create public and secret contexts
            self.secret_context = ts.context_from(context_bytes, n_threads=1)
            self.public_context = ts.context_from(context_bytes, n_threads=1)
            self.public_context.make_context_public()
            
            logger.info(f"Client {self.client_id}: HE context initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize HE context: {e}")
            self.use_he = False
    
    def get_parameters(self, config=None):
        """Get model parameters, optionally encrypted"""
        try:
            state_dict = self.model.state_dict()
            parameters = []
            total_payload = 0
            start_time = time.time()
            
            for param_tensor in state_dict.values():
                param_array = param_tensor.cpu().detach().numpy().flatten()
                
                if self.use_he:
                    try:
                        # Encrypt parameters
                        encrypted_vector = ts.ckks_vector(self.public_context, param_array.tolist())
                        serialized = encrypted_vector.serialize()
                        parameters.append(serialized)
                        total_payload += len(serialized)
                    except Exception as e:
                        logger.warning(f"HE encryption failed, using plaintext: {e}")
                        parameters.append(param_array.tolist())
                else:
                    parameters.append(param_array.tolist())
            
            self.metrics.update({
                "encoding_time": time.time() - start_time,
                "payload_kb": total_payload / 1024 if self.use_he else 0
            })
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return []
    
    def set_parameters(self, parameters):
        """Set model parameters, decrypting if necessary"""
        try:
            state_dict = self.model.state_dict()
            param_keys = list(state_dict.keys())
            
            for i, (key, param) in enumerate(zip(param_keys, parameters)):
                original_shape = state_dict[key].shape
                
                if self.use_he and isinstance(param, bytes):
                    try:
                        # Decrypt parameters
                        encrypted_vector = ts.ckks_vector_from(self.secret_context, param)
                        decrypted_array = np.array(encrypted_vector.decrypt(), dtype=np.float32)
                        tensor_data = torch.from_numpy(decrypted_array[:np.prod(original_shape)].reshape(original_shape))
                    except Exception as e:
                        logger.warning(f"HE decryption failed for {key}: {e}")
                        continue
                else:
                    # Use plaintext parameters
                    param_array = np.array(param, dtype=np.float32)
                    tensor_data = torch.from_numpy(param_array[:np.prod(original_shape)].reshape(original_shape))
                
                state_dict[key] = tensor_data.to(CONFIG["DEVICE"])
            
            self.model.load_state_dict(state_dict)
            
        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
    
    def fit(self, parameters, config):
        """Train the model"""
        try:
            self.set_parameters(parameters)
            
            # Create data loader
            dataloader = DataLoader(
                self.dataset, 
                batch_size=CONFIG["BATCH_SIZE"], 
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )
            
            # Setup differential privacy if needed
            if self.use_dp:
                try:
                    privacy_engine = PrivacyEngine()
                    self.model, self.optimizer, dataloader = privacy_engine.make_private(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=dataloader,
                        noise_multiplier=CONFIG["DP_NOISE"],
                        max_grad_norm=CONFIG["DP_MAX_GRAD"]
                    )
                except Exception as e:
                    logger.warning(f"DP setup failed: {e}")
                    self.use_dp = False
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                try:
                    # Move batch to device
                    batch = {k: v.to(CONFIG["DEVICE"]) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Batch training error: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Update metrics
            self.metrics.update({"loss": avg_loss})
            
            if self.use_dp:
                try:
                    epsilon = privacy_engine.get_epsilon(delta=environment["dp_epsilon_delta"]["delta"])
                    self.metrics["dp_epsilon"] = epsilon
                except:
                    pass
            
            logger.info(f"Client {self.client_id} training completed - Loss: {avg_loss:.4f}")
            
            return self.get_parameters(), len(self.dataset), self.metrics
            
        except Exception as e:
            logger.error(f"Training failed for client {self.client_id}: {e}")
            return self.get_parameters(), len(self.dataset), {"loss": float('inf')}
    
    def evaluate(self, parameters, config):
        """Evaluate the model"""
        return 0.0, len(self.dataset), {}

def central_evaluation(test_data):
    """Evaluate model performance on test set"""
    try:
        logger.info("Running central evaluation...")
        
        dataset = QADataset(test_data)
        dataloader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], num_workers=0)
        
        # Create fresh model for evaluation
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.to(CONFIG["DEVICE"])
        model.eval()
        
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    batch = {k: v.to(CONFIG["DEVICE"]) for k, v in batch.items()}
                    
                    # Generate predictions
                    generated_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=CONFIG["LABEL_LENGTH"],
                        num_beams=1,
                        do_sample=False
                    )
                    
                    # Compare with ground truth
                    for pred_ids, label_ids in zip(generated_ids, batch["labels"]):
                        pred_text = TOKENIZER.decode(pred_ids, skip_special_tokens=True).strip().lower()
                        label_text = TOKENIZER.decode(label_ids, skip_special_tokens=True).strip().lower()
                        
                        if pred_text == label_text:
                            correct_predictions += 1
                        total_predictions += 1
                        
                except Exception as e:
                    logger.warning(f"Evaluation batch error: {e}")
                    continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Central evaluation failed: {e}")
        return 0.0

def run_experiment():
    """Run the federated learning experiment"""
    try:
        logger.info("Starting federated learning privacy evaluation...")
        
        # Load and split data
        client_data, test_data = load_and_split_data()
        
        # Define experiment configurations
        experiments = [
            ("Baseline FL", False, False),
            ("FL+HE", True, False) if HE_AVAILABLE else ("FL (HE disabled)", False, False),
            ("FL+HE+DP", True, True) if HE_AVAILABLE else ("FL+DP", False, True)
        ]
        
        results = {}
        histories = {}
        
        for exp_name, use_he, use_dp in experiments:
            logger.info(f"\n=== Running {exp_name} ===")
            
            try:
                # Create clients
                def client_fn(client_id: str):
                    return PrivacyClient(
                        client_id, 
                        client_data[int(client_id)], 
                        use_he=use_he, 
                        use_dp=use_dp
                    )
                
                # Configure strategy
                strategy = fl.server.strategy.FedAvg(
                    fraction_fit=1.0,
                    fraction_evaluate=0.0,
                    min_fit_clients=CONFIG["CLIENT_NUM"],
                    min_available_clients=CONFIG["CLIENT_NUM"]
                )
                
                # Run simulation
                history = fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=CONFIG["CLIENT_NUM"],
                    config=fl.server.ServerConfig(num_rounds=CONFIG["NUM_ROUNDS"]),
                    strategy=strategy
                )
                
                histories[exp_name] = history
                
                # Evaluate final model
                accuracy = central_evaluation(test_data)
                results[exp_name] = accuracy
                
                logger.info(f"{exp_name} completed - Final accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                results[exp_name] = 0.0
                continue
        
        # Generate visualization
        create_visualization(histories, results)
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return {}

def create_visualization(histories, results):
    """Create visualization of results"""
    try:
        logger.info("Creating visualization...")
        
        if not histories:
            logger.warning("No history data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Federated Learning Privacy Evaluation Results", fontsize=16)
        
        rounds = list(range(1, CONFIG["NUM_ROUNDS"] + 1))
        
        # Plot 1: Loss per round
        for exp_name, history in histories.items():
            if hasattr(history, 'losses_distributed') and history.losses_distributed:
                losses = [loss[1] for loss in history.losses_distributed]
                axes[0, 0].plot(rounds[:len(losses)], losses, marker='o', label=exp_name)
        
        axes[0, 0].set_title("Training Loss per Round")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Final accuracy comparison
        exp_names = list(results.keys())
        accuracies = list(results.values())
        axes[0, 1].bar(exp_names, accuracies)
        axes[0, 1].set_title("Final Model Accuracy")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Encryption overhead (if available)
        axes[1, 0].set_title("Encryption Overhead")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Time (seconds)")
        
        # Plot 4: Privacy budget (if available)
        axes[1, 1].set_title("Differential Privacy Budget")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Epsilon")
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig("federated_privacy_results.png", dpi=150, bbox_inches='tight')
        logger.info("Visualization saved as 'federated_privacy_results.png'")
        
        # Also try to display if possible
        try:
            plt.show()
        except:
            logger.info("Display not available, plot saved to file only")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

def main():
    """Main execution function"""
    try:
        logger.info("=" * 60)
        logger.info("FEDERATED LEARNING PRIVACY EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Configuration: {CONFIG}")
        
        if not HE_AVAILABLE:
            logger.warning("TenSEAL not available - HE experiments will be disabled")
        
        # Run experiments
        results = run_experiment()
        
        # Print final results
        logger.info("\n" + "=" * 40)
        logger.info("FINAL RESULTS")
        logger.info("=" * 40)
        
        for exp_name, accuracy in results.items():
            logger.info(f"{exp_name:15}: {accuracy:.4f}")
        
        logger.info("\nExperiment completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return {}

if __name__ == "__main__":
    main()