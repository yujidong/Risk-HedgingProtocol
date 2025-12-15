"""
TEE Performance Benchmark - LSTM Model Evaluation with SGX
===========================================================
This script implements a performance benchmark for running LSTM traffic
prediction models in TEE (Trusted Execution Environment) using Intel SGX
via Gramine.

The benchmark measures:
1. Model loading time
2. Inference latency
3. Signature generation overhead
4. Total end-to-end time
5. Memory usage

Author: Experiment Team
Date: December 2025
"""

import json
import time
import hashlib
import numpy as np
import psutil
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

# ============================================================================
# Configuration
# ============================================================================

# Mock TEE Private Key (in production, this would be securely provisioned)
MOCK_TEE_PRIVATE_KEY = "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# Model configuration
WINDOW_SIZE = 24
HIDDEN_SIZE = 256
NUM_LOCATIONS = 170
BATCH_SIZE = 32

# Device configuration (CPU only for SGX compatibility)
device = torch.device('cpu')

# ============================================================================
# LSTM Model Definition
# ============================================================================

class TrafficLSTM(nn.Module):
    """
    LSTM model for traffic prediction.
    Same architecture as the main experiment for consistency.
    """
    
    def __init__(self, input_size, hidden_size=256, output_size=170, dropout_rate=0.2):
        super(TrafficLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, output_size)
    
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, (hidden, _) = self.lstm2(lstm_out1)
        out = hidden[-1]
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_pems08_data(data_path='data/PEMS08.npz', sample_size=None):
    """
    Load and preprocess PEMS08 traffic data with proper hour-based aggregation.
    
    Args:
        data_path: Path to PEMS08.npz file
        sample_size: If None, use full dataset; otherwise limit timesteps
    
    Returns:
        Tuple of (X, Y, scaler_X, scaler_Y)
    """
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure PEMS08.npz is in the data/ directory")
        return None, None, None, None
    
    data = np.load(data_path)
    traffic_data = data['data']  # Shape: (num_samples, num_nodes, num_features)
    
    print(f"Raw data shape: {traffic_data.shape}")
    
    # Limit data if specified
    if sample_size is not None:
        traffic_data = traffic_data[:sample_size]
        print(f"Using subset: {traffic_data.shape}")
    
    # Convert to DataFrame for hour-based aggregation (like successful experiment)
    import pandas as pd
    data_dict = []
    for timestep in range(traffic_data.shape[0]):
        for location in range(traffic_data.shape[1]):
            data_dict.append({
                "timestep": timestep + 1,
                "location": location,
                "flow": traffic_data[timestep][location][0],
                "occupy": traffic_data[timestep][location][1],
                "speed": traffic_data[timestep][location][2]
            })
    
    traffic = pd.DataFrame(data_dict)
    
    # Create time series dataset with hour-based aggregation
    def create_dataset(location, WINDOW_SIZE=24):
        location_current = traffic[traffic["location"] == location].reset_index()
        location_current["hour"] = ((location_current["timestep"] - 1) // 12)
        grouped = location_current.groupby("hour").mean().reset_index()
        grouped['day'] = (grouped['hour'] // 24) % 7
        grouped['hour'] %= 24
        
        one_hot_hour = pd.get_dummies(grouped['hour'])
        one_hot_hour = one_hot_hour.add_prefix('hour_')
        
        hour_grouped = pd.concat([grouped[["occupy", "flow", "speed"]], one_hot_hour], axis=1)
        hour_grouped = np.array(hour_grouped)
        
        X, Y = [], []
        for i in range(len(hour_grouped) - WINDOW_SIZE):
            X.append(hour_grouped[i:(i + WINDOW_SIZE)][::-1])
            Y.append(hour_grouped[i + WINDOW_SIZE, 0])
        
        return X, Y
    
    # Create dataset for all locations
    X, Y = [], []
    for location in range(170):
        a, b = create_dataset(location, WINDOW_SIZE=WINDOW_SIZE)
        X.append(a)
        Y.append(b)
    
    X = np.moveaxis(X, 0, -1)
    Y = np.moveaxis(Y, 0, -1)
    
    print(f"After aggregation - X: {X.shape}, Y: {Y.shape}")
    
    # Normalize
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    original_shape = X.shape
    X = scaler_X.fit_transform(X.reshape(X.shape[0] * X.shape[1], -1)) \
                 .reshape(original_shape)
    Y = scaler_Y.fit_transform(Y)
    
    # Reshape from 4D to 3D for LSTM: (samples, timesteps, features*locations)
    X = X.reshape(X.shape[0], X.shape[1], -1)
    
    print(f"Processed data - X: {X.shape}, Y: {Y.shape}")
    
    return X, Y, scaler_X, scaler_Y


def create_dummy_data(num_samples=100):
    """
    Create dummy data for testing when PEMS08 is not available.
    """
    print("Creating dummy data for testing...")
    
    X = np.random.randn(num_samples, WINDOW_SIZE, NUM_LOCATIONS * 3)
    Y = np.random.randn(num_samples, NUM_LOCATIONS)
    
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    X = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    Y = scaler_Y.fit_transform(Y)
    
    return X, Y, scaler_X, scaler_Y


# ============================================================================
# Model Operations
# ============================================================================

def train_model(model, train_X, train_Y, epochs=50, batch_size=64, lr=0.001):
    """
    Train the LSTM model.
    
    Args:
        model: PyTorch model to train
        train_X: Training input data
        train_Y: Training target data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    
    Returns:
        Trained model and training time
    """
    print(f"Training model for {epochs} epochs...")
    start_time = time.time()
    
    # Convert to tensors
    train_X_tensor = torch.FloatTensor(train_X).to(device)
    train_Y_tensor = torch.FloatTensor(train_Y).to(device)
    
    # Create data loader
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_X_tensor, train_Y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_X, batch_Y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    model.eval()
    return model, training_time


def load_or_create_model(model_path='models/traffic_lstm.pth', input_size=None, 
                         train_X=None, train_Y=None, force_retrain=False):
    """
    Load pre-trained model or create and train a new one.
    
    Args:
        model_path: Path to saved model
        input_size: Input feature size (required if creating new model)
        train_X: Training data for new model
        train_Y: Training labels for new model
        force_retrain: If True, retrain even if model exists
    
    Returns:
        Loaded PyTorch model and training time (0 if loaded from disk)
    """
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading model from: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model, 0.0
    else:
        print(f"Creating new model (input_size={input_size})")
        if input_size is None:
            raise ValueError("input_size required to create new model")
        
        model = TrafficLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, 
                           output_size=NUM_LOCATIONS)
        
        # Train the model if training data provided
        training_time = 0.0
        if train_X is not None and train_Y is not None:
            model, training_time = train_model(model, train_X, train_Y, epochs=50)
        
        model.eval()
        
        # Save for future use
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)
        print(f"Model saved to: {model_path}")
        
        return model, training_time


def run_inference(model, test_X, batch_size=32):
    """
    Run model inference on test data.
    
    Args:
        model: PyTorch model
        test_X: Test input data
        batch_size: Batch size for inference
    
    Returns:
        Tuple of (predictions, inference_time)
    """
    model.eval()
    
    test_X_tensor = torch.FloatTensor(test_X).to(device)
    num_samples = len(test_X)
    all_predictions = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = test_X_tensor[i:i + batch_size]
            predictions = model(batch).cpu().numpy()
            all_predictions.append(predictions)
    
    inference_time = time.time() - start_time
    
    predictions = np.vstack(all_predictions)
    
    return predictions, inference_time


# ============================================================================
# TEE Signature Operations
# ============================================================================

def generate_signature(order_id, utility_score, nonce):
    """
    Generate cryptographic signature using TEE private key.
    
    Args:
        order_id: Order identifier (uint256)
        utility_score: Utility score as integer (scaled by 1e18)
        nonce: Nonce bytes (32 bytes)
    
    Returns:
        Tuple of (signature_hex, signer_address, signing_time)
    """
    start_time = time.time()
    
    # Create message hash (mimicking Solidity's keccak256)
    message_hash = Web3.solidity_keccak(
        ['uint256', 'uint256', 'bytes32'],
        [order_id, utility_score, nonce]
    )
    
    # Sign with private key
    account = Account.from_key(MOCK_TEE_PRIVATE_KEY)
    signed_message = account.unsafe_sign_hash(message_hash)
    
    signature = signed_message.signature.hex()
    signing_time = time.time() - start_time
    
    return signature, account.address, signing_time


# ============================================================================
# Benchmarking
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def run_benchmark(num_iterations=10, data_size=100, batch_size=32, low_memory=False):
    """
    Run complete benchmark measuring all performance metrics.
    
    Args:
        num_iterations: Number of benchmark iterations
        data_size: Number of test samples
        batch_size: Batch size for inference (smaller = less memory)
        low_memory: Enable low-memory mode (process data in smaller chunks)
    
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*80)
    print("TEE PERFORMANCE BENCHMARK - LSTM Model")
    print("="*80)
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print(f"Data size: {data_size}")
    print(f"Batch size: {batch_size}")
    print(f"Low memory mode: {low_memory}")
    print("="*80 + "\n")
    
    results = {
        'config': {
            'num_iterations': num_iterations,
            'data_size': data_size,
            'batch_size': batch_size,
            'low_memory': low_memory,
            'window_size': WINDOW_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        },
        'timings': {
            'data_loading': [],
            'model_loading': [],
            'inference': [],
            'signature_generation': [],
            'total_end_to_end': []
        },
        'metrics': {
            'r2_score': [],
            'rmse': []
        },
        'memory': {
            'initial_mb': get_memory_usage(),
            'peak_mb': get_memory_usage(),
            'final_mb': get_memory_usage()
        }
    }
    
    # Load data once
    print("Loading data...")
    data_load_start = time.time()
    # Use full dataset (None) or limit timesteps for faster testing
    X, Y, scaler_X, scaler_Y = load_pems08_data(sample_size=None)
    
    if X is None:
        print("Using dummy data instead...")
        X, Y, scaler_X, scaler_Y = create_dummy_data(num_samples=data_size)
    
    data_load_time = time.time() - data_load_start
    print(f"Data loaded in {data_load_time:.3f}s\n")
    
    # Split data into train/test
    train_size = int(0.8 * len(X))
    train_X, test_X = X[:train_size], X[train_size:]
    train_Y, test_Y = Y[:train_size], Y[train_size:]
    print(f"Train size: {len(train_X)}, Test size: {len(test_X)}\n")
    
    # Load or train model
    print("Loading/Training model...")
    model_load_start = time.time()
    input_size = X.shape[-1]
    model, training_time = load_or_create_model(
        input_size=input_size, 
        train_X=train_X, 
        train_Y=train_Y,
        force_retrain=False  # Set to True to retrain existing model
    )
    model_load_time = time.time() - model_load_start
    print(f"Model ready in {model_load_time:.3f}s\n")
    
    # Use test set for evaluation
    X, Y = test_X, test_Y
    print(f"Running benchmark on test set ({len(X)} samples)\n")
    
    # Run benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...\n")
    
    for iteration in range(num_iterations):
        print(f"[Iteration {iteration + 1}/{num_iterations}]")
        
        # Measure memory before iteration
        mem_before = get_memory_usage()
        
        # ===== STEP 1: Model Inference =====
        inference_start = time.time()
        predictions, _ = run_inference(model, X, batch_size=batch_size)
        inference_time = time.time() - inference_start
        
        # Calculate metrics
        predictions_original = scaler_Y.inverse_transform(predictions)
        Y_original = scaler_Y.inverse_transform(Y)
        
        r2 = r2_score(Y_original, predictions_original)
        rmse = np.sqrt(mean_squared_error(Y_original, predictions_original))
        
        print(f"  Inference: {inference_time:.3f}s (R2={r2:.4f}, RMSE={rmse:.2f})")
        
        # ===== STEP 2: Signature Generation =====
        # Simulate signing result for each prediction
        # In practice, you'd sign aggregated results
        order_id = iteration + 1
        # Clamp R2 to [0, 1] range and scale to integer
        utility_score = int(max(0, min(1, r2)) * 10**18)
        nonce = hashlib.sha256(f"nonce_{iteration}".encode()).digest()
        
        signature, signer, signing_time = generate_signature(order_id, utility_score, nonce)
        
        print(f"  Signature: {signing_time:.3f}s (Signer: {signer[:10]}...)")
        
        # ===== STEP 3: Total Time =====
        total_time = inference_time + signing_time
        
        print(f"  Total: {total_time:.3f}s")
        
        # Measure memory after iteration
        mem_after = get_memory_usage()
        results['memory']['peak_mb'] = max(results['memory']['peak_mb'], mem_after)
        
        # Store results
        results['timings']['inference'].append(inference_time)
        results['timings']['signature_generation'].append(signing_time)
        results['timings']['total_end_to_end'].append(total_time)
        results['metrics']['r2_score'].append(r2)
        results['metrics']['rmse'].append(rmse)
        
        print()
    
    # Store one-time measurements
    results['timings']['data_loading'] = [data_load_time]
    results['timings']['model_loading'] = [model_load_time]
    results['timings']['model_training'] = [training_time] if training_time > 0 else []
    results['memory']['final_mb'] = get_memory_usage()
    
    # Calculate statistics
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nData Loading: {data_load_time:.3f}s")
    print(f"Model Loading/Training: {model_load_time:.3f}s")
    if training_time > 0:
        print(f"  (Training time: {training_time:.1f}s)")
    print(f"\nInference Time:")
    print(f"  Mean: {np.mean(results['timings']['inference']):.3f}s")
    print(f"  Std:  {np.std(results['timings']['inference']):.3f}s")
    print(f"  Min:  {np.min(results['timings']['inference']):.3f}s")
    print(f"  Max:  {np.max(results['timings']['inference']):.3f}s")
    
    print(f"\nSignature Generation Time:")
    print(f"  Mean: {np.mean(results['timings']['signature_generation']):.3f}s")
    print(f"  Std:  {np.std(results['timings']['signature_generation']):.3f}s")
    print(f"  Min:  {np.min(results['timings']['signature_generation']):.3f}s")
    print(f"  Max:  {np.max(results['timings']['signature_generation']):.3f}s")
    
    print(f"\nTotal End-to-End Time:")
    print(f"  Mean: {np.mean(results['timings']['total_end_to_end']):.3f}s")
    print(f"  Std:  {np.std(results['timings']['total_end_to_end']):.3f}s")
    print(f"  Min:  {np.min(results['timings']['total_end_to_end']):.3f}s")
    print(f"  Max:  {np.max(results['timings']['total_end_to_end']):.3f}s")
    
    print(f"\nModel Performance:")
    print(f"  R2 Score: {np.mean(results['metrics']['r2_score']):.4f} +/- {np.std(results['metrics']['r2_score']):.4f}")
    print(f"  RMSE:     {np.mean(results['metrics']['rmse']):.2f} +/- {np.std(results['metrics']['rmse']):.2f}")
    
    print(f"\nMemory Usage:")
    print(f"  Initial: {results['memory']['initial_mb']:.2f} MB")
    print(f"  Peak:    {results['memory']['peak_mb']:.2f} MB")
    print(f"  Final:   {results['memory']['final_mb']:.2f} MB")
    
    print("\n" + "="*80 + "\n")
    
    return results


def save_results(results, output_file='tee_benchmark_results.json'):
    """Save benchmark results to JSON file."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TEE Performance Benchmark for LSTM Model')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--data-size', type=int, default=100,
                       help='Number of test samples (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (smaller = less memory, default: 32)')
    parser.add_argument('--low-memory', action='store_true',
                       help='Enable low-memory mode (process data in smaller chunks)')
    parser.add_argument('--output', type=str, default='results/tee_benchmark_results.json',
                       help='Output file path (default: results/tee_benchmark_results.json)')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        num_iterations=args.iterations,
        data_size=args.data_size,
        batch_size=args.batch_size,
        low_memory=args.low_memory
    )
    
    # Save results
    save_results(results, args.output)
    
    print("Benchmark complete!")
