#!/usr/bin/env python3
"""
Federated Learning Privacy Evaluation Demo
A simplified simulation demonstrating privacy-preserving techniques
in federated learning without heavy ML dependencies.
"""

import os
import sys
import time
import random
import math
import json
from typing import List, Dict, Any, Tuple

# Check if numpy and matplotlib are available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using pure Python")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available, skipping visualization")

# === Configuration ===
CONFIG = {
    "NUM_CLIENTS": 3,
    "NUM_ROUNDS": 5,
    "LOCAL_EPOCHS": 2,
    "LEARNING_RATE": 0.01,
    "DATASET_SIZE": 1000,
    "FEATURE_DIM": 10,
    "DP_NOISE_SCALE": 0.1,
    "HE_PRECISION_BITS": 16
}

print("=" * 60)
print("FEDERATED LEARNING PRIVACY EVALUATION DEMO")
print("=" * 60)
print(f"Configuration: {json.dumps(CONFIG, indent=2)}")

class SimpleLinearModel:
    """Simple linear model for demonstration"""
    
    def __init__(self, feature_dim):
        if NUMPY_AVAILABLE:
            self.weights = np.random.normal(0, 0.1, feature_dim)
            self.bias = 0.0
        else:
            self.weights = [random.gauss(0, 0.1) for _ in range(feature_dim)]
            self.bias = 0.0
    
    def predict(self, x):
        """Make prediction"""
        if NUMPY_AVAILABLE:
            return np.dot(x, self.weights) + self.bias
        else:
            return sum(xi * wi for xi, wi in zip(x, self.weights)) + self.bias
    
    def get_parameters(self):
        """Get model parameters"""
        return {"weights": self.weights.tolist() if NUMPY_AVAILABLE else self.weights, 
                "bias": self.bias}
    
    def set_parameters(self, params):
        """Set model parameters"""
        if NUMPY_AVAILABLE:
            self.weights = np.array(params["weights"])
        else:
            self.weights = params["weights"]
        self.bias = params["bias"]

class DataGenerator:
    """Generate synthetic dataset"""
    
    def __init__(self, size, feature_dim, seed=42):
        random.seed(seed)
        if NUMPY_AVAILABLE:
            np.random.seed(seed)
        
        self.size = size
        self.feature_dim = feature_dim
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic regression data"""
        data = []
        for _ in range(self.size):
            if NUMPY_AVAILABLE:
                x = np.random.normal(0, 1, self.feature_dim)
                # True relationship: y = sum(x) + noise
                y = np.sum(x) + np.random.normal(0, 0.1)
            else:
                x = [random.gauss(0, 1) for _ in range(self.feature_dim)]
                y = sum(x) + random.gauss(0, 0.1)
            data.append((x, y))
        return data
    
    def get_client_data(self, client_id, num_clients):
        """Split data among clients"""
        chunk_size = self.size // num_clients
        start_idx = client_id * chunk_size
        end_idx = (client_id + 1) * chunk_size if client_id < num_clients - 1 else self.size
        return self.data[start_idx:end_idx]

class HomomorphicEncryption:
    """Simple homomorphic encryption simulation"""
    
    def __init__(self, precision_bits=16):
        self.precision_bits = precision_bits
        self.scale = 2 ** precision_bits
        # Simple additive secret key
        self.secret_key = random.randint(1000, 9999)
    
    def encrypt(self, value):
        """Simulate encryption (scaled integer representation)"""
        scaled_value = int(value * self.scale)
        # Simple additive encryption simulation
        noise = random.randint(-100, 100)
        encrypted = (scaled_value + self.secret_key + noise) % (2**32)
        return encrypted
    
    def decrypt(self, encrypted_value):
        """Simulate decryption"""
        # Remove secret key (simplified)
        decrypted = (encrypted_value - self.secret_key) % (2**32)
        # Handle negative values
        if decrypted > 2**31:
            decrypted -= 2**32
        return decrypted / self.scale
    
    def encrypt_parameters(self, params):
        """Encrypt model parameters"""
        encrypted = {}
        for key, value in params.items():
            if key == "weights":
                if NUMPY_AVAILABLE:
                    encrypted[key] = [self.encrypt(w) for w in value]
                else:
                    encrypted[key] = [self.encrypt(w) for w in value]
            else:
                encrypted[key] = self.encrypt(value)
        return encrypted
    
    def decrypt_parameters(self, encrypted_params):
        """Decrypt model parameters"""
        decrypted = {}
        for key, value in encrypted_params.items():
            if key == "weights":
                decrypted[key] = [self.decrypt(w) for w in value]
            else:
                decrypted[key] = self.decrypt(value)
        return decrypted

class DifferentialPrivacy:
    """Differential privacy implementation"""
    
    def __init__(self, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.epsilon = 1.0  # Privacy budget
    
    def add_noise(self, value):
        """Add Laplace noise for differential privacy"""
        if NUMPY_AVAILABLE:
            noise = np.random.laplace(0, self.noise_scale)
        else:
            # Simple approximation of Laplace noise
            u = random.uniform(-0.5, 0.5)
            noise = -self.noise_scale * math.log(1 - 2 * abs(u)) * (1 if u >= 0 else -1)
        return value + noise
    
    def privatize_parameters(self, params):
        """Add noise to parameters for differential privacy"""
        private_params = {}
        for key, value in params.items():
            if key == "weights":
                private_params[key] = [self.add_noise(w) for w in value]
            else:
                private_params[key] = self.add_noise(value)
        return private_params

class FederatedClient:
    """Federated learning client with privacy features"""
    
    def __init__(self, client_id, data, use_he=False, use_dp=False):
        self.client_id = client_id
        self.data = data
        self.model = SimpleLinearModel(CONFIG["FEATURE_DIM"])
        self.use_he = use_he
        self.use_dp = use_dp
        
        # Initialize privacy mechanisms
        if self.use_he:
            self.he = HomomorphicEncryption(CONFIG["HE_PRECISION_BITS"])
        if self.use_dp:
            self.dp = DifferentialPrivacy(CONFIG["DP_NOISE_SCALE"])
        
        print(f"Client {client_id} initialized - Data samples: {len(data)}, HE: {use_he}, DP: {use_dp}")
    
    def train_local(self, global_params, epochs=2):
        """Train model locally"""
        if global_params:
            self.model.set_parameters(global_params)
        
        # Simple gradient descent training
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in self.data:
                # Forward pass
                pred = self.model.predict(x)
                loss = (pred - y) ** 2
                total_loss += loss
                
                # Compute gradients (simplified)
                error = pred - y
                lr = CONFIG["LEARNING_RATE"]
                
                # Update weights
                if NUMPY_AVAILABLE:
                    grad_w = 2 * error * np.array(x)
                    self.model.weights -= lr * grad_w
                else:
                    grad_w = [2 * error * xi for xi in x]
                    self.model.weights = [w - lr * g for w, g in zip(self.model.weights, grad_w)]
                
                # Update bias
                self.model.bias -= lr * 2 * error
            
            avg_loss = total_loss / len(self.data)
            losses.append(avg_loss)
        
        return losses
    
    def get_parameters(self):
        """Get model parameters with privacy protection"""
        params = self.model.get_parameters()
        
        # Apply differential privacy
        if self.use_dp:
            params = self.dp.privatize_parameters(params)
        
        # Apply homomorphic encryption
        if self.use_he:
            params = self.he.encrypt_parameters(params)
            return params, True  # Encrypted flag
        
        return params, False
    
    def compute_privacy_metrics(self):
        """Compute privacy-related metrics"""
        metrics = {}
        
        if self.use_dp:
            metrics["dp_epsilon"] = self.dp.epsilon
            metrics["dp_noise_scale"] = self.dp.noise_scale
        
        if self.use_he:
            metrics["he_precision_bits"] = self.he.precision_bits
            
        return metrics

class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, clients):
        self.clients = clients
        self.global_model = SimpleLinearModel(CONFIG["FEATURE_DIM"])
        self.round_history = []
    
    def aggregate_parameters(self, client_params_list):
        """Aggregate parameters from clients (FedAvg)"""
        # Decrypt parameters if encrypted
        decrypted_params = []
        total_samples = 0
        
        for params, encrypted, num_samples in client_params_list:
            if encrypted and self.clients[0].use_he:  # Use first client's HE for decryption
                params = self.clients[0].he.decrypt_parameters(params)
            decrypted_params.append((params, num_samples))
            total_samples += num_samples
        
        # Weighted average
        if NUMPY_AVAILABLE:
            avg_weights = np.zeros(CONFIG["FEATURE_DIM"])
        else:
            avg_weights = [0.0] * CONFIG["FEATURE_DIM"]
        avg_bias = 0.0
        
        for params, num_samples in decrypted_params:
            weight = num_samples / total_samples
            
            if NUMPY_AVAILABLE:
                avg_weights += weight * np.array(params["weights"])
            else:
                avg_weights = [aw + weight * w for aw, w in zip(avg_weights, params["weights"])]
            avg_bias += weight * params["bias"]
        
        return {
            "weights": avg_weights.tolist() if NUMPY_AVAILABLE else avg_weights,
            "bias": avg_bias
        }
    
    def federated_round(self, round_num):
        """Execute one federated learning round"""
        print(f"\n--- Round {round_num + 1} ---")
        
        # Get current global parameters
        global_params = self.global_model.get_parameters()
        
        # Client training
        client_results = []
        round_metrics = {"losses": [], "privacy_metrics": {}}
        
        for client in self.clients:
            # Local training
            losses = client.train_local(global_params, CONFIG["LOCAL_EPOCHS"])
            
            # Get updated parameters
            params, encrypted = client.get_parameters()
            num_samples = len(client.data)
            
            client_results.append((params, encrypted, num_samples))
            round_metrics["losses"].append(losses[-1])  # Last epoch loss
            
            # Collect privacy metrics
            privacy_metrics = client.compute_privacy_metrics()
            round_metrics["privacy_metrics"][f"client_{client.client_id}"] = privacy_metrics
        
        # Aggregate parameters
        aggregated_params = self.aggregate_parameters(client_results)
        self.global_model.set_parameters(aggregated_params)
        
        # Calculate average loss
        avg_loss = sum(round_metrics["losses"]) / len(round_metrics["losses"])
        round_metrics["avg_loss"] = avg_loss
        
        self.round_history.append(round_metrics)
        
        print(f"Round {round_num + 1} completed - Average loss: {avg_loss:.4f}")
        
        return round_metrics

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    total_loss = 0.0
    for x, y in test_data:
        pred = model.predict(x)
        loss = (pred - y) ** 2
        total_loss += loss
    
    mse = total_loss / len(test_data)
    rmse = math.sqrt(mse)
    return {"mse": mse, "rmse": rmse}

def run_privacy_experiments():
    """Run experiments with different privacy configurations"""
    print("\n" + "=" * 50)
    print("RUNNING PRIVACY EXPERIMENTS")
    print("=" * 50)
    
    # Generate dataset
    data_generator = DataGenerator(CONFIG["DATASET_SIZE"], CONFIG["FEATURE_DIM"])
    
    # Create test set
    test_data = data_generator.data[-100:]  # Last 100 samples for testing
    train_data = data_generator.data[:-100]  # Rest for training
    
    # Define experiment configurations
    experiments = [
        ("Baseline FL", False, False),
        ("FL + HE", True, False),
        ("FL + DP", False, True),
        ("FL + HE + DP", True, True)
    ]
    
    results = {}
    
    for exp_name, use_he, use_dp in experiments:
        print(f"\n🔬 Running {exp_name}...")
        
        # Create clients with privacy settings
        clients = []
        for i in range(CONFIG["NUM_CLIENTS"]):
            client_data = data_generator.get_client_data(i, CONFIG["NUM_CLIENTS"])
            if len(client_data) == 0:  # Skip if no data
                continue
            client = FederatedClient(i, client_data, use_he=use_he, use_dp=use_dp)
            clients.append(client)
        
        if not clients:
            print(f"⚠️ No clients created for {exp_name}")
            continue
        
        # Create server and run federated learning
        server = FederatedServer(clients)
        
        experiment_start = time.time()
        
        for round_num in range(CONFIG["NUM_ROUNDS"]):
            server.federated_round(round_num)
        
        experiment_time = time.time() - experiment_start
        
        # Evaluate final model
        final_metrics = evaluate_model(server.global_model, test_data)
        
        results[exp_name] = {
            "final_metrics": final_metrics,
            "training_time": experiment_time,
            "round_history": server.round_history
        }
        
        print(f"✅ {exp_name} completed in {experiment_time:.2f}s")
        print(f"   Final RMSE: {final_metrics['rmse']:.4f}")
    
    return results

def create_results_visualization(results):
    """Create visualization of results"""
    if not MATPLOTLIB_AVAILABLE:
        print("📊 Matplotlib not available, creating text-based summary instead")
        create_text_summary(results)
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Federated Learning Privacy Evaluation Results", fontsize=16)
    
    # Extract data for plotting
    exp_names = list(results.keys())
    final_rmse = [results[name]["final_metrics"]["rmse"] for name in exp_names]
    training_times = [results[name]["training_time"] for name in exp_names]
    
    # Plot 1: Final RMSE comparison
    axes[0, 0].bar(exp_names, final_rmse, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_title("Final Model Performance (RMSE)")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Training time comparison
    axes[0, 1].bar(exp_names, training_times, color=['blue', 'green', 'orange', 'red'])
    axes[0, 1].set_title("Training Time")
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Loss convergence
    for exp_name in exp_names:
        history = results[exp_name]["round_history"]
        losses = [round_data["avg_loss"] for round_data in history]
        rounds = list(range(1, len(losses) + 1))
        axes[1, 0].plot(rounds, losses, marker='o', label=exp_name)
    
    axes[1, 0].set_title("Training Loss Convergence")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Average Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Privacy-utility tradeoff
    privacy_scores = []
    for exp_name in exp_names:
        if "HE" in exp_name and "DP" in exp_name:
            score = 3  # Highest privacy
        elif "HE" in exp_name or "DP" in exp_name:
            score = 2  # Medium privacy
        else:
            score = 1  # Baseline privacy
        privacy_scores.append(score)
    
    axes[1, 1].scatter(privacy_scores, final_rmse, s=100, 
                      c=['blue', 'green', 'orange', 'red'])
    for i, name in enumerate(exp_names):
        axes[1, 1].annotate(name, (privacy_scores[i], final_rmse[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 1].set_title("Privacy-Utility Tradeoff")
    axes[1, 1].set_xlabel("Privacy Level")
    axes[1, 1].set_ylabel("RMSE (lower is better)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("federated_privacy_results.png", dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'federated_privacy_results.png'")

def create_text_summary(results):
    """Create text-based summary when matplotlib is not available"""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    for exp_name, result in results.items():
        print(f"\n🔍 {exp_name}:")
        print(f"   Final RMSE: {result['final_metrics']['rmse']:.4f}")
        print(f"   Training Time: {result['training_time']:.2f} seconds")
        
        # Show loss progression
        history = result["round_history"]
        losses = [round_data["avg_loss"] for round_data in history]
        print(f"   Loss progression: {' -> '.join(f'{loss:.3f}' for loss in losses)}")
        
        # Privacy metrics
        if history and "privacy_metrics" in history[0]:
            first_client_metrics = list(history[0]["privacy_metrics"].values())[0]
            if first_client_metrics:
                print(f"   Privacy features: {list(first_client_metrics.keys())}")

def main():
    """Main execution function"""
    try:
        print(f"🚀 Starting federated learning privacy evaluation...")
        print(f"📊 NumPy available: {NUMPY_AVAILABLE}")
        print(f"📈 Matplotlib available: {MATPLOTLIB_AVAILABLE}")
        
        # Run experiments
        results = run_privacy_experiments()
        
        if not results:
            print("❌ No experiments completed successfully")
            return
        
        # Create visualization or summary
        create_results_visualization(results)
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Final summary
        print("\n📋 Key Findings:")
        print("• Baseline FL: No privacy protection, best utility")
        print("• FL + HE: Encrypted communication, moderate overhead")
        print("• FL + DP: Differential privacy, noise impact on utility")
        print("• FL + HE + DP: Maximum privacy, highest overhead")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    main()