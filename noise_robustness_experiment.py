"""
Core Noise Robustness Experiment
================================
This script implements the complete experimental framework for evaluating
LSTM model robustness under two critical noise conditions:
1. Random Drop: Simulating missing sensor data
2. Data Scarcity with Drift: Limited training data under temporal distribution shift

The experiment quantifies both performance degradation and uncertainty amplification,
demonstrating that data quality issues primarily impact model reliability rather than
average predictive capacity.

Author: Yuji Dong
Date: December 2025
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Random seed for reproducibility
def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

# Global cache for base data (loaded once)
_base_data_cache = None

def load_base_data(data_path='input/pems-dataset/data/PEMS08/PEMS08.npz'):
    """
    Load base traffic data from disk (cached).
    Uses the same preprocessing as the successful high_variance_experiment.
    """
    global _base_data_cache
    
    if _base_data_cache is not None:
        return _base_data_cache
    
    print(f"Loading base data from: {data_path}")
    data = np.load(data_path)
    traffic_data = data['data']  # Shape: (num_samples, num_nodes, num_features)
    
    # Convert to DataFrame for hour-based aggregation
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
        a, b = create_dataset(location, WINDOW_SIZE=24)
        X.append(a)
        Y.append(b)
    
    X = np.moveaxis(X, 0, -1)
    Y = np.moveaxis(Y, 0, -1)
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(X))
    train_data_X = X[:train_size].copy()
    train_data_Y = Y[:train_size].copy()
    test_data_X = X[train_size:].copy()
    test_data_Y = Y[train_size:].copy()
    
    _base_data_cache = (train_data_X, train_data_Y, test_data_X, test_data_Y)
    print(f"  Base data loaded: Train={train_data_X.shape}, Test={test_data_X.shape}")
    
    return _base_data_cache


def apply_random_drop_noise(data, drop_rate):
    """Apply random drop noise by randomly setting values to zero."""
    if drop_rate == 0:
        return data.copy()
    
    noisy_data = data.copy()
    mask = np.random.rand(*data.shape) < drop_rate
    noisy_data[mask] = 0
    return noisy_data


def apply_drift_noise(data, drift_intensity):
    """
    Apply drift noise by adding linearly increasing bias.
    Data should already be normalized (0-1 range from MinMaxScaler).
    """
    if drift_intensity == 0:
        return data.copy()
    
    noisy_data = data.copy()
    num_samples = len(data)
    
    # Create linearly increasing drift
    # For normalized data, add proportional drift
    drift_factor = np.linspace(0, drift_intensity, num_samples)
    # Reshape to match data dimensions (works for both 3D and 4D)
    drift_factor = drift_factor.reshape(-1, *([1] * (len(data.shape) - 1)))
    
    noisy_data += drift_factor * noisy_data
    return noisy_data


def reduce_training_data(data, data_ratio):
    """Reduce training data by randomly sampling."""
    if data_ratio >= 1.0:
        return data
    
    num_samples = int(len(data) * data_ratio)
    indices = np.random.choice(len(data), num_samples, replace=False)
    return data[indices]


def prepare_dataset(noise_config, base_train_X, base_train_Y, base_test_X, base_test_Y):
    """
    Prepare dataset with specified noise configuration.
    Following the successful approach from high_variance_experiment:
    1. Reduce training data if needed
    2. Normalize with MinMaxScaler
    3. Apply noise AFTER normalization
    
    Args:
        noise_config: Dict with keys 'type', 'intensity', 'train_data_ratio'
        base_train_X, base_train_Y: Original training data
        base_test_X, base_test_Y: Original test data
    
    Returns:
        Tuple of (train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y)
    """
    noise_type = noise_config['type']
    intensity = noise_config['intensity']
    data_ratio = noise_config['train_data_ratio']
    
    # Copy data
    train_X = base_train_X.copy()
    train_Y = base_train_Y.copy()
    test_X = base_test_X.copy()
    test_Y = base_test_Y.copy()
    
    # 1. Reduce training data if needed (data scarcity)
    if data_ratio < 1.0:
        reduced_size = max(50, int(len(train_X) * data_ratio))
        train_X = train_X[:reduced_size]
        train_Y = train_Y[:reduced_size]
    
    # 2. Normalize FIRST (like the successful experiment)
    # train_X shape: (samples, timesteps, features, locations) -> flatten to (samples*timesteps, features*locations)
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    
    original_shape = train_X.shape
    train_X = scaler_X.fit_transform(train_X.reshape(train_X.shape[0] * train_X.shape[1], -1)) \
                       .reshape(original_shape)
    test_X = scaler_X.transform(test_X.reshape(test_X.shape[0] * test_X.shape[1], -1)) \
                      .reshape(test_X.shape)
    train_Y = scaler_Y.fit_transform(train_Y)
    test_Y = scaler_Y.transform(test_Y)
    
    # 3. Apply noise AFTER normalization (this is the key!)
    if noise_type == 'random_drop' and intensity > 0:
        train_X = apply_random_drop_noise(train_X, intensity)
    elif noise_type == 'drift' and intensity > 0:
        train_X = apply_drift_noise(train_X, intensity)
    
    # 4. Reshape from 4D to 3D for LSTM: (samples, timesteps, features*locations)
    # Original shape: (samples, timesteps, features, locations)
    # Target shape: (samples, timesteps, features*locations)
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], -1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], -1)
    
    return train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y


# ============================================================================
# LSTM Model Definition
# ============================================================================

class TrafficLSTM(nn.Module):
    """
    LSTM model for traffic forecasting.
    Architecture matches the successful high_variance_experiment:
    - 2-layer LSTM with hidden_size=256
    - Additional FC layer (256 units)
    - Dropout for regularization
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
# Training Functions
# ============================================================================

def train_single_model(train_X, train_Y, test_X, test_Y, scaler_Y, 
                       epochs=50, batch_size=64, lr=0.001):
    """Train a single LSTM model and evaluate."""
    
    # Convert to tensors
    train_X_tensor = torch.FloatTensor(train_X)
    train_Y_tensor = torch.FloatTensor(train_Y)
    test_X_tensor = torch.FloatTensor(test_X)
    test_Y_tensor = torch.FloatTensor(test_Y)
    
    # Create data loaders
    train_dataset = TensorDataset(train_X_tensor, train_Y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = train_X.shape[-1]
    output_size = train_Y.shape[-1]
    model = TrafficLSTM(input_size=input_size, hidden_size=256, output_size=output_size)
    
    # Dummy forward pass to initialize LSTM (workaround for PyTorch 2.10 bug)
    with torch.no_grad():
        dummy_input = torch.randn(1, train_X.shape[1], input_size)
        _ = model(dummy_input)
    
    # Move to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_X_tensor = test_X_tensor.to(device)
        predictions = model(test_X_tensor).cpu().numpy()
    
    # Inverse transform predictions
    predictions_original = scaler_Y.inverse_transform(predictions)
    test_Y_original = scaler_Y.inverse_transform(test_Y)
    
    # Calculate metrics
    r2 = r2_score(test_Y_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(test_Y_original, predictions_original))
    
    return r2, rmse


def train_multiple_models_parallel(data_configs, n_repeats, epochs, batch_train_size=12):
    """
    Train multiple models in parallel (interleaved batch processing).
    
    Args:
        data_configs: List of data configuration tuples
        n_repeats: Number of models to train
        epochs: Training epochs
        batch_train_size: Number of models to train in parallel
    """
    all_results = []
    
    for batch_start in range(0, n_repeats, batch_train_size):
        batch_end = min(batch_start + batch_train_size, n_repeats)
        batch_size = batch_end - batch_start
        
        print(f"  Training models {batch_start+1}-{batch_end}/{n_repeats} in parallel...")
        
        # Initialize models for this batch
        models = []
        optimizers = []
        train_loaders = []
        
        for i in range(batch_size):
            train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y = data_configs[batch_start + i]
            
            # Create model
            input_size = train_X.shape[-1]
            output_size = train_Y.shape[-1]
            model = TrafficLSTM(input_size=input_size, hidden_size=256, output_size=output_size)
            
            # Dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, train_X.shape[1], input_size)
                _ = model(dummy_input)
            
            model = model.to(device)
            models.append(model)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizers.append(optimizer)
            
            # Create data loader
            train_X_tensor = torch.FloatTensor(train_X)
            train_Y_tensor = torch.FloatTensor(train_Y)
            train_dataset = TensorDataset(train_X_tensor, train_Y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            train_loaders.append(train_loader)
        
        # Training loop (interleaved)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for model_idx in range(batch_size):
                models[model_idx].train()
                for batch_X, batch_Y in train_loaders[model_idx]:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)
                    
                    outputs = models[model_idx](batch_X)
                    loss = criterion(outputs, batch_Y)
                    
                    optimizers[model_idx].zero_grad()
                    loss.backward()
                    optimizers[model_idx].step()
        
        # Evaluate all models in this batch
        for i in range(batch_size):
            model = models[i]
            train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y = data_configs[batch_start + i]
            
            model.eval()
            with torch.no_grad():
                test_X_tensor = torch.FloatTensor(test_X).to(device)
                predictions = model(test_X_tensor).cpu().numpy()
            
            predictions_original = scaler_Y.inverse_transform(predictions)
            test_Y_original = scaler_Y.inverse_transform(test_Y)
            
            r2 = r2_score(test_Y_original, predictions_original)
            rmse = np.sqrt(mean_squared_error(test_Y_original, predictions_original))
            
            all_results.append({'r2': r2, 'rmse': rmse})
    
    return all_results


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_noise_robustness_experiment(experiment_configs, n_repeats=20, epochs=50, 
                                   batch_train_size=12, output_file='experiment_results.json',
                                   use_timestamp=True):
    """
    Run the complete noise robustness experiment.
    
    Args:
        experiment_configs: List of dicts with 'type', 'intensity', 'train_data_ratio'
        n_repeats: Number of repetitions per configuration
        epochs: Training epochs
        batch_train_size: Parallel batch size
        output_file: Output JSON file path (will add timestamp if use_timestamp=True)
        use_timestamp: If True, append timestamp to output filename to avoid overwriting
    """
    
    # Generate timestamped filename if requested
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(output_file).stem
        extension = Path(output_file).suffix
        output_file = f"output/data/{base_name}_{timestamp}{extension}"
    else:
        output_file = f"output/data/{output_file}"
    
    # Ensure output directory exists
    Path("output/data").mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NOISE ROBUSTNESS EXPERIMENT")
    print("="*80)
    print(f"Total configurations: {len(experiment_configs)}")
    print(f"Repeats per configuration: {n_repeats}")
    print(f"Total models to train: {len(experiment_configs) * n_repeats}")
    print(f"Device: {device}")
    print(f"Output file: {output_file}")
    print("="*80)
    
    # Load base data
    base_train_X, base_train_Y, base_test_X, base_test_Y = load_base_data()
    
    # Store results
    results = {}
    total_start_time = time.time()
    
    for config_idx, config in enumerate(experiment_configs):
        config_key = f"{config['type']}_{config['intensity']:.2f}_data{config['train_data_ratio']:.2f}"
        
        print(f"\n[{config_idx+1}/{len(experiment_configs)}] Configuration: {config_key}")
        print(f"  Type: {config['type']}, Intensity: {config['intensity']}, Data Ratio: {config['train_data_ratio']}")
        
        # Pre-generate all data variations
        print(f"  Generating {n_repeats} data variations...")
        data_configs = []
        for i in range(n_repeats):
            # Use session timestamp to make different runs produce different results
            # while keeping same-session results reproducible
            import hashlib
            session_seed = int(hashlib.md5(session_timestamp.encode()).hexdigest()[:8], 16) % 1000000
            set_seed(session_seed + i)
            train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y = prepare_dataset(
                config, base_train_X, base_train_Y, base_test_X, base_test_Y
            )
            data_configs.append((train_X, train_Y, test_X, test_Y, scaler_X, scaler_Y))
        
        # Train models in parallel
        config_start_time = time.time()
        model_results = train_multiple_models_parallel(data_configs, n_repeats, epochs, batch_train_size)
        config_time = time.time() - config_start_time
        
        # Aggregate results
        r2_scores = [r['r2'] for r in model_results]
        rmse_scores = [r['rmse'] for r in model_results]
        
        results[config_key] = {
            'r2_score': {
                'mean': float(np.mean(r2_scores)),
                'std': float(np.std(r2_scores)),
                'variance': float(np.var(r2_scores)),
                'all_values': [float(x) for x in r2_scores],
                'n_successful': len(r2_scores)
            },
            'rmse': {
                'mean': float(np.mean(rmse_scores)),
                'std': float(np.std(rmse_scores)),
                'all_values': [float(x) for x in rmse_scores]
            },
            'training_time_seconds': float(config_time)
        }
        
        print(f"  Results: R²={np.mean(r2_scores):.4f}±{np.std(r2_scores):.4f}, "
              f"RMSE={np.mean(rmse_scores):.4f}±{np.std(rmse_scores):.4f}")
        print(f"  Time: {config_time:.1f}s")
    
    total_time = time.time() - total_start_time
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETE!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return results


# ============================================================================
# Experiment Configurations
# ============================================================================

def get_random_drop_configs():
    """Generate Random Drop experiment configurations."""
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    configs = []
    for rate in drop_rates:
        configs.append({
            'type': 'random_drop',
            'intensity': rate,
            'train_data_ratio': 1.0
        })
    return configs


def get_data_scarcity_configs():
    """Generate Data Scarcity with Drift experiment configurations."""
    drift_intensities = [0.30, 0.40, 0.50, 0.60]
    data_ratios = [1.0, 0.7, 0.5, 0.3]
    configs = []
    for drift in drift_intensities:
        for ratio in data_ratios:
            configs.append({
                'type': 'drift',
                'intensity': drift,
                'train_data_ratio': ratio
            })
    return configs


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Determine which experiment to run
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        print("Usage: python noise_robustness_experiment.py [random_drop|data_scarcity|both]")
        experiment_type = 'both'
    
    # Generate timestamp for this experiment session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SESSION: {session_timestamp}")
    print(f"{'='*80}\n")
    
    if experiment_type in ['random_drop', 'both']:
        print("\n" + "="*80)
        print("EXPERIMENT 1: RANDOM DROP")
        print("="*80)
        random_drop_configs = get_random_drop_configs()
        run_noise_robustness_experiment(
            random_drop_configs,
            n_repeats=20,
            epochs=50,
            batch_train_size=12,
            output_file='random_drop_results.json',
            use_timestamp=True
        )
    
    if experiment_type in ['data_scarcity', 'both']:
        print("\n" + "="*80)
        print("EXPERIMENT 2: DATA SCARCITY WITH DRIFT")
        print("="*80)
        data_scarcity_configs = get_data_scarcity_configs()
        run_noise_robustness_experiment(
            data_scarcity_configs,
            n_repeats=20,
            epochs=50,
            batch_train_size=12,
            output_file='data_scarcity_results.json',
            use_timestamp=True
        )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Run visualization script to generate figures:")
    print("  python visualize_results.py")
    print("="*80)
