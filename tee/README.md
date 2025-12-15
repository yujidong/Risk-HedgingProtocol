# TEE Performance Benchmark for LSTM Models

Comprehensive testing framework for measuring LSTM model performance overhead in Trusted Execution Environments (TEE) using Intel SGX.

## Test Results Summary

**Environment**: Azure DC2s_v3 VM (Intel SGX, 8GB EPC)
**Dataset**: PEMS08 traffic data (17,856 timesteps)
**Model**: 2-layer LSTM (hidden_size=256)

| Metric | Native Docker | Gramine SGX | Overhead |
|--------|---------------|-------------|----------|
| Inference Time | 547ms +/- 18ms | 642ms +/- 5ms | +17.3% |
| Model Accuracy (R2) | 0.8063 | 0.8063 | Identical |
| Memory Peak | 2047 MB | 2420 MB | +18.2% |
| Data Loading | 20.2s | 35.7s | +76.6% |

**Key Finding**: Only 17% inference overhead in real SGX TEE with identical accuracy.

---

## Package Contents

`
tee-benchmark-package/
|-- tee_benchmark.py              # Main benchmark script
|-- Dockerfile.gramine             # SGX Docker image configuration
|-- tee_benchmark.manifest.template # Gramine SGX manifest
|-- run_cloud_test.sh             # Automated test runner (Native + SGX)
|-- requirements.txt              # Python dependencies
|-- create_azure_vm.ps1           # Azure VM creation script
|-- redeploy_all.ps1              # Deployment automation script
|-- data/
|   \-- PEMS08.npz               # Traffic dataset (user must provide)
|-- models/
|   \-- traffic_lstm.pth         # Trained LSTM model (user must provide)
|-- results/                      # Test results output directory
\-- docs/
    |-- README.md                # This file
    |-- SETUP.md                 # Setup instructions
    \-- ARCHITECTURE.md          # Technical details
`

---

## Quick Start

### Prerequisites

1. Azure Account with SGX-capable VM access (DC-series)
2. Data Files (not included, must be provided):
   - data/PEMS08.npz - Traffic dataset
   - models/traffic_lstm.pth - Pre-trained LSTM model
3. Local Requirements:
   - Azure CLI (az)
   - PowerShell (Windows) or Bash (Linux/macOS)
   - SSH client

### Step 1: Setup Data Files

Place your data files in the correct locations:

`ash
# Required files (user must provide):
./data/PEMS08.npz           # Shape: (17856, 170, 3)
./models/traffic_lstm.pth   # PyTorch model state_dict
`

**Data Format Requirements**:
- PEMS08.npz must contain: data array with shape (timesteps, locations, features)
- Model must be a 2-layer LSTM compatible with the data dimensions
- See tee_benchmark.py for data processing details

### Step 2: Create Azure VM

`powershell
# Edit create_azure_vm.ps1 to set your parameters
# Then run:
.\create_azure_vm.ps1
`

**VM Requirements**:
- Instance Type: Standard_DC2s_v3 (or higher DC-series)
- OS: Ubuntu 20.04/22.04 LTS
- Disk: 30GB minimum
- SGX: Must have SGX support with 8GB+ EPC

### Step 3: Deploy and Test

`powershell
# Deploy everything to Azure VM
.\redeploy_all.ps1

# SSH to VM and run tests
ssh azureuser@<VM_IP>
chmod +x run_cloud_test.sh
./run_cloud_test.sh
`

### Step 4: Retrieve Results

`powershell
# Download results from VM
scp azureuser@<VM_IP>:~/results/*.json ./results/

# Stop VM to avoid charges
az vm deallocate --resource-group web3 --name tee-benchmark-vm
`

---

## Test Outputs

Each test run generates JSON files with detailed metrics:

`json
{
  "config": {
    "num_iterations": 10,
    "device": "cpu",
    "timestamp": "2025-12-15T07:13:42"
  },
  "timings": {
    "data_loading": [20.22],
    "inference": [0.549, 0.539, ...],
    "signature": [0.013, 0.008, ...]
  },
  "model_performance": {
    "r2_score": 0.8063,
    "rmse": 0.02
  },
  "memory": {
    "initial_mb": 620.82,
    "peak_mb": 2047.48
  }
}
`

---

## Configuration Files

### 1. Dockerfile.gramine

**Key Settings**:
`dockerfile
# CRITICAL: Use CPU-only PyTorch for SGX compatibility
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Use real SGX hardware (not simulation)
ENTRYPOINT ["gramine-sgx", "tee_benchmark"]
`

### 2. tee_benchmark.manifest.template

**Key Settings**:
`	oml
sgx.enclave_size = "8G"  # Must be power of 2: 1G, 2G, 4G, 8G, 16G
sgx.max_threads = 32
sgx.debug = true         # Set to false for production
`

**Enclave Size Guidelines**:
- 2GB: Fails with full dataset (memory error)
- 4GB: Marginal for 17K samples
- 8GB: Recommended for full PEMS08 dataset
- 16GB: Only if VM has 16GB+ SGX EPC

### 3. run_cloud_test.sh

Automated test sequence:
1. Native Docker Test - Baseline performance
2. Gramine SGX Test - TEE performance
3. Analysis - Automatic comparison report

---

## Technical Details

### SGX Configuration Validation

**How to verify real SGX (not simulation)**:

1. Check logs for:
   `
   sgx.debug = true (this is a debug enclave)
   Emulating a raw syscall instruction
   `

2. Docker must mount SGX devices:
   `ash
   --device=/dev/sgx_enclave
   --device=/dev/sgx_provision
   -v /var/run/aesmd:/var/run/aesmd
   `

3. Dockerfile must use:
   `dockerfile
   ENTRYPOINT ["gramine-sgx", "tee_benchmark"]  # NOT gramine-direct
   `

### Performance Characteristics

**Expected Overhead**:
- Pure computation (inference): 10-30%
- Memory-intensive operations: 50-100%
- I/O operations: 100-300%

**Your Results**:
- Inference: 17.3% (within expected range)
- Memory: 18.2% (within expected range)
- Data loading: 76.6% (expected for I/O)

---

## Important Notes

### 1. Data Files Not Included

This package does NOT include:
- data/PEMS08.npz - You must provide your own dataset
- models/traffic_lstm.pth - You must provide your trained model

**Reason**: Large file sizes and potential licensing restrictions.

### 2. Costs

Azure DC2s_v3 pricing:
- Running: ~\.248/hour
- Stopped (deallocated): ~\.005/day (disk only)
- Typical test: ~\.10 (25 minutes)

Always deallocate VM when not in use:
`ash
az vm deallocate --resource-group web3 --name tee-benchmark-vm
`

### 3. Security

**For production use**:
1. Set sgx.debug = false in manifest
2. Remove sgx.allowed_files (use trusted_files only)
3. Implement remote attestation
4. Use production signing keys

**Current configuration is for research/testing only.**

### 4. Python Version

- Native container: Python 3.10
- Gramine container: Python 3.12
- Why different: Gramine base image uses 3.12

Both versions produce identical results.

---

## Troubleshooting

### Common Issues

1. "Unable to allocate memory" in SGX
   - Solution: Increase sgx.enclave_size to 8GB or higher
   - Must be power of 2: 1G, 2G, 4G, 8G, 16G

2. "libtorch_cuda.so: failed to map segment"
   - Solution: Ensure CPU-only PyTorch is installed
   - Check: --index-url https://download.pytorch.org/whl/cpu

3. No performance overhead (nearly identical)
   - Problem: Running in simulation mode (gramine-direct)
   - Solution: Use gramine-sgx in Dockerfile ENTRYPOINT

4. "Data file not found" in Docker
   - Solution: Ensure cd /app && before running Python
   - Files must be copied into Docker image, not just mounted

5. SSH permission denied
   - Solution: chmod +x run_cloud_test.sh on VM
   - Or run: bash run_cloud_test.sh

---

## References

### Academic Papers

- Intel SGX Performance: Intel Official Documentation
- ML in TEEs: "Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware"
- TEE Overhead Studies: Various USENIX/IEEE papers show 10-30% for compute-intensive tasks

### Related Technologies

- Gramine: https://gramineproject.io/
- Intel SGX: https://www.intel.com/sgx
- Azure Confidential Computing: https://azure.microsoft.com/solutions/confidential-compute/

---

## Citation

If you use this benchmark in your research, please cite:

`
TEE Performance Benchmark for LSTM Models
Testing Framework for Intel SGX with Gramine
Azure DC2s_v3 (Ice Lake SGX)
Dataset: PeMS08 Traffic Data (17,856 timesteps)
Result: 17.3% inference overhead with identical accuracy
`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review docs/ARCHITECTURE.md for technical details
3. Verify all prerequisites are met
4. Check Azure VM has SGX enabled

---

## License

Please ensure compliance with:
- Your dataset license (PEMS08)
- Your model license
- Intel SGX SDK license
- Gramine license (LGPL v3)
- Azure terms of service

---

**Last Updated**: December 15, 2025
**Tested Environment**: Azure DC2s_v3, Ubuntu 22.04, Gramine latest, PyTorch 2.5.1
