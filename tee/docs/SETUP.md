# Setup Guide

## Prerequisites Installation

### Windows (PowerShell)

\\\powershell
# Install Azure CLI
winget install Microsoft.AzureCLI

# Verify installation
az --version

# Login to Azure
az login
\\\

### Linux/macOS

\\\ash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login
\\\

## Data Preparation

### 1. Obtain PEMS08 Dataset

You need to provide your own PEMS08.npz file with the following format:

\\\python
# Expected structure:
import numpy as np
data = np.load('data/PEMS08.npz')
print(data['data'].shape)  # Should be: (17856, 170, 3)
# Dimensions: (timesteps, sensors, features)
# Features: [flow, occupancy, speed]
\\\

### 2. Obtain or Train LSTM Model

Your model must be compatible with the data:

\\\python
# Model architecture (from tee_benchmark.py):
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        # 2-layer LSTM with specified hidden size
        # Output: predictions for each sensor
\\\

Save model state_dict:
\\\python
torch.save(model.state_dict(), 'models/traffic_lstm.pth')
\\\

## Azure Configuration

### 1. Create Resource Group

\\\ash
az group create --name web3 --location eastus
\\\

### 2. Verify SGX Availability

\\\ash
# Check if DC-series VMs are available in your region
az vm list-skus --location eastus --size Standard_DC --output table
\\\

### 3. Set Default Subscription (if needed)

\\\ash
az account list --output table
az account set --subscription "<your-subscription-id>"
\\\

## Local Testing (Optional)

Before deploying to Azure, you can test locally (without SGX):

\\\ash
# Install dependencies
pip install -r requirements.txt

# Run benchmark locally
python tee_benchmark.py --iterations 5 --output results/local_test.json
\\\

## File Checklist

Before running redeploy_all.ps1, ensure these files exist:

- [ ] data/PEMS08.npz (user-provided, ~17.7 MB)
- [ ] models/traffic_lstm.pth (user-provided, ~21 MB)
- [ ] tee_benchmark.py
- [ ] Dockerfile.gramine
- [ ] tee_benchmark.manifest.template
- [ ] run_cloud_test.sh
- [ ] requirements.txt

## Validation

### Verify PEMS08.npz format:

\\\python
import numpy as np
data = np.load('data/PEMS08.npz')
print("Keys:", list(data.keys()))
print("Shape:", data['data'].shape)
assert data['data'].shape == (17856, 170, 3), "Incorrect data shape!"
\\\

### Verify model compatibility:

\\\python
import torch
state_dict = torch.load('models/traffic_lstm.pth', map_location='cpu')
print("Model keys:", list(state_dict.keys()))
# Should contain: lstm.weight_ih_l0, lstm.weight_hh_l0, fc.weight, fc.bias, etc.
\\\

## Next Steps

Once setup is complete:
1. Run create_azure_vm.ps1 to create VM
2. Run redeploy_all.ps1 to deploy code
3. SSH to VM and execute ./run_cloud_test.sh
4. Download results and deallocate VM
