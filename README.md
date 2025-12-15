# Risk-Hedging Equity Protocol for IoT Data Trading

A complete blockchain-based implementation of the **Trustworthy Data Equity Protocol** combining smart contracts, TEE-secured computation, and AI-powered data analysis for secure IoT data trading.

> **📄 Paper**: This repository contains the official implementation for the paper *"Trustworthy Data Equity: A Retrospective Risk-Hedging Protocol for High-Entropy IoT Data Assets"* submitted to **IEEE Internet of Things Journal**.

## Overview

This project implements an end-to-end decentralized data trading system with three integrated components:

### 🔗 1. Blockchain Smart Contracts
- **Smart Contract Settlement**: Ethereum-compatible contracts handle order creation, escrow, and atomic payment
- **Multi-role Architecture**: Separate accounts for Owner, Buyer, Seller, and TEE Signer
- **Testnet Deployment**: Production-ready deployment on Sepolia, Arbitrum, and Optimism testnets
- **Comprehensive Benchmarks**: Gas cost and performance analysis across L1/L2 networks

### 🔒 2. TEE (Trusted Execution Environment)
- **Intel SGX Integration**: Secure model inference in hardware-protected enclaves via Gramine
- **Performance Benchmarking**: Native vs TEE overhead analysis (~17% inference overhead)
- **Cloud Deployment**: Automated Azure VM setup scripts for SGX-capable infrastructure
- **Cryptographic Attestation**: ECDSA signature generation for on-chain verification

### 🤖 3. AI/ML Data Analysis
- **LSTM Traffic Prediction**: Deep learning models trained on PeMS traffic dataset
- **Noise Robustness Testing**: Validates protocol resilience to sensor noise and data quality issues
- **Game Theory Experiments**: Data scarcity and random drop scenario analysis
- **Automated Visualization**: Comprehensive plotting and results analysis tools

## 🏗️ Project Structure

```
Risk-HedgingProtocol/
├── contracts/                    # 📜 Smart Contracts
│   └── protocol.sol              #    DataEquityProtocol (Solidity)
│
├── test/                         # 🧪 Contract Testing
│   ├── DataEquityProtocol.test.js          # Functional tests
│   ├── DataEquityProtocol.benchmark.js     # L1 performance benchmarks
│   └── DataEquityProtocol.benchmark-simple.js
│
├── scripts/                      # 🛠️ Utility Scripts
│   ├── check-all-balances.js    # Multi-account balance checker
│   ├── compare_benchmarks.py    # Benchmark results comparison
│   └── run_all_benchmarks.ps1   # Automated test runner
│
├── tee/                          # 🔒 TEE Component (NEW!)
│   ├── tee_benchmark.py         # Main TEE benchmark script
│   ├── Dockerfile.gramine       # SGX container configuration
│   ├── tee_benchmark.manifest.template # Gramine SGX manifest
│   ├── create_azure_vm.ps1      # Azure SGX VM provisioning
│   ├── run_cloud_test.sh        # Automated TEE testing
│   ├── data/                    # PEMS08 dataset (shared with ML)
│   ├── models/                  # Trained LSTM models
│   ├── results/                 # TEE benchmark results
│   └── docs/                    # TEE architecture documentation
│
├── input/                        # 📊 ML Datasets
│   └── pems-dataset/            # PeMS traffic dataset (PEMS03/04/07/08)
│
├── output/                       # 📈 Experiment Results
│   ├── data/                    # ML experiment results (JSON)
│   ├── figures/                 # Visualization plots (PNG)
│   └── benchmark/               # Blockchain benchmark results
│
├── noise_robustness_experiment.py # 🤖 Main ML experiments
├── game_theory_data_scarcity.py   # Game theory: data scarcity
├── game_theory_random_drop.py     # Game theory: random drops
├── visualize_results.py           # Results visualization
│
├── hardhat.config.ts             # Hardhat configuration (3 networks)
├── ACCOUNTS.md                   # Multi-account setup guide
├── DEPLOYMENT.md                 # Deployment instructions
└── README.md                     # This file
```

## Smart Contract Features

### DataEquityProtocol Contract

**Core Functions:**
- `createOrder()`: Buyer locks funds in escrow with pricing parameters
- `settleTransaction()`: TEE-signed utility score triggers atomic payment
- `refund()`: Timeout protection for buyers
- `setTEESigner()`: Admin function to update TEE public key

**Pricing Model:**
```
Final Payment = p_base + α * k * u

Where:
- p_base: Base fee (fixed)
- α: Equity share coefficient (0 ≤ α ≤ 1)
- k: Utility-to-money conversion factor
- u: Data utility score (0 ≤ u ≤ 1, TEE-verified)
```

**Security:**
- ECDSA signature verification for TEE attestation
- ReentrancyGuard protection
- Ownable access control
- Nonce-based replay attack prevention

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js** 22+ and npm 11+ (blockchain development)
- **Python** 3.9+ with conda (ML experiments)
- **Docker** (optional, for TEE testing)
- **Azure Account** (optional, for SGX cloud deployment)

### 1️⃣ Blockchain Setup

```bash
# Install dependencies
npm install

# Compile contracts
npx hardhat compile

# Run functional tests
npx hardhat test

# Run performance benchmarks (requires testnet ETH)
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
```

### 2️⃣ ML/AI Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate risk-hedging

# Run noise robustness experiments
python noise_robustness_experiment.py

# Run game theory analysis
python game_theory_data_scarcity.py
python game_theory_random_drop.py

# Generate visualizations
python visualize_results.py
```

### 3️⃣ TEE Setup (Advanced)

```bash
cd tee/

# Option A: Local testing with dummy data
python tee_benchmark.py --mode native --iterations 10

# Option B: Cloud SGX deployment
# 1. Create Azure VM with SGX support
.\create_azure_vm.ps1

# 2. Deploy and run tests
.\redeploy_all.ps1

# 3. SSH to VM and execute
ssh azureuser@<VM_IP>
chmod +x run_cloud_test.sh
./run_cloud_test.sh

# Results will be in tee/results/*.json
```

## 📊 Complete Workflow Example

### End-to-End Data Trading Simulation

**Step 1: Generate Training Data**
```bash
# ML experiments produce utility scores and model performance
python noise_robustness_experiment.py
# Output: output/data/*.json
```

**Step 2: Deploy Smart Contract**
```bash
# Set up accounts and deploy to testnet
npx hardhat keystore set SEPOLIA_PRIVATE_KEY --dev
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network sepolia
# Contract address: 0xE0aa880da6822A26C946f9417F7F6380FDf9799F (example)
```

**Step 3: TEE Utility Evaluation**
```bash
cd tee/
# TEE validates data and signs utility score
python tee_benchmark.py --mode native
# Output: Utility score (0.80) + ECDSA signature
```

**Step 4: On-Chain Settlement**
```bash
# Buyer creates order with locked funds
# TEE submits signed utility score
# Smart contract verifies signature and executes payment
npx hardhat test test/DataEquityProtocol.test.js --network sepolia
# ✅ Order settled: Seller receives payment based on utility
```

**Step 5: Analyze Results**
```bash
# Compare blockchain benchmarks
python scripts/compare_benchmarks.py

# Visualize all experimental data
python visualize_results.py
```

## 📖 Detailed Usage

## 📖 Detailed Usage

### Blockchain Component

#### Deploy to Public Testnet

```bash
# Configure accounts (Owner, Buyer, Seller, TEE Signer)
# See ACCOUNTS.md for multi-account setup guide
npx hardhat keystore set SEPOLIA_PRIVATE_KEY
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_2
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_3
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_4

# Check balances
npx hardhat run scripts/check-all-balances.js --network sepolia

# Deploy contract
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network sepolia

# Run comprehensive benchmarks
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
```

**Benchmark Results** (Sepolia L1 @ 3 gwei):
- Deployment: 21,000 gas (~$0.06)
- Order Creation: 273,077 gas (~$2.46)
- Settlement: 73,708 gas (~$0.63)
- Refund: 44,745 gas (~$0.41)

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

### AI/ML Component

#### LSTM Noise Robustness Experiments

```bash
# Train and evaluate LSTM models with noise injection
python noise_robustness_experiment.py

# Key parameters:
# - Noise levels: 0%, 10%, 20%, 30%, 40%, 50%
# - Datasets: PEMS03, PEMS04, PEMS07, PEMS08
# - Model: 2-layer LSTM (hidden_size=256)

# Output: output/data/*_results.json
# Metrics: MAE, RMSE, R2 score per noise level
```

#### Game Theory Experiments

```bash
# Data scarcity scenarios (50-100% data availability)
python game_theory_data_scarcity.py

# Random drop scenarios (probability-based data loss)
python game_theory_random_drop.py

# Generate comparison plots
python visualize_results.py
```

### TEE Component

#### Local Testing (Without SGX Hardware)

```bash
cd tee/

# Create dummy data for testing
python tee_benchmark.py --mode native --iterations 10 --use-dummy-data

# Output: results/native_*.json
```

#### Cloud SGX Deployment (Production)

**Prerequisites**:
- Azure account with quota for DC-series VMs
- Azure CLI installed and authenticated
- SSH key pair configured

**Deployment Steps**:

```bash
cd tee/

# 1. Provision Azure SGX VM
.\create_azure_vm.ps1
# Creates: Standard_DC2s_v3 VM with 8GB SGX EPC

# 2. Deploy code and dependencies
.\redeploy_all.ps1
# Uploads: tee_benchmark.py, data/, models/, configs

# 3. SSH to VM and run tests
ssh azureuser@<VM_IP>

# 4. Run native baseline
docker run --rm -v $(pwd):/app tee-benchmark:native python tee_benchmark.py

# 5. Run SGX enclave test
docker run --device /dev/sgx_enclave --device /dev/sgx_provision \
  -v $(pwd):/app tee-benchmark:sgx gramine-sgx python

# 6. Retrieve results
exit
scp azureuser@<VM_IP>:~/results/*.json ./results/

# 7. Stop VM to avoid charges
az vm deallocate --resource-group web3 --name tee-benchmark-vm
```

**TEE Performance Results**:
- Native Inference: 547ms ± 18ms
- SGX Inference: 642ms ± 5ms (+17.3% overhead)
- Model Accuracy: Identical (R² = 0.8063)
- Memory Overhead: +18.2%

See [tee/docs/ARCHITECTURE.md](tee/docs/ARCHITECTURE.md) for technical details.

## 🧪 Experimental Results

## 🧪 Experimental Results Summary

### 1. Blockchain Performance (Sepolia L1 Testnet)

| Operation | Gas Used | Cost (@3000 gwei) | Time |
|-----------|----------|-------------------|------|
| Contract Deployment | 21,000 | $0.06 | ~11s |
| Order Creation | 273,077 | $2.46 | ~2s |
| Order Settlement | 73,708 | $0.63 | ~2s |
| Refund | 44,745 | $0.41 | ~2s |
| **Complete Trade** | **346,785** | **$3.09** | **~4s** |

💡 **L2 Recommendation**: Deploying to Arbitrum/Optimism can reduce costs by 95% (~$0.15/trade)

### 2. TEE Performance (Azure DC2s_v3 + Intel SGX)

| Metric | Native Docker | Gramine SGX | Overhead |
|--------|---------------|-------------|----------|
| Inference Time | 547ms ± 18ms | 642ms ± 5ms | **+17.3%** |
| Model Accuracy (R²) | 0.8063 | 0.8063 | **Identical** |
| Memory Usage | 2047 MB | 2420 MB | +18.2% |
| Signature Generation | 13ms ± 3ms | 8ms ± 2ms | -38% |

🔒 **Key Finding**: Only **17% inference overhead** in real SGX hardware with **identical accuracy**

### 3. AI/ML Noise Robustness

**LSTM Model Performance under Noise:**

| Noise Level | MAE | RMSE | R² Score | Utility Score |
|-------------|-----|------|----------|---------------|
| 0% (Clean) | 2.87 | 4.12 | 0.953 | 1.00 |
| 10% | 3.02 | 4.31 | 0.945 | 0.95 |
| 20% | 3.24 | 4.58 | 0.932 | 0.89 |
| 30% | 3.51 | 4.91 | 0.915 | 0.81 |
| 40% | 3.89 | 5.34 | 0.891 | 0.72 |
| 50% | 4.42 | 5.98 | 0.852 | 0.61 |

✅ **Protocol Stability**: Maintains reliable utility scores up to 40% noise level

**Game Theory Results**:
- Data scarcity: Linear degradation from 100% → 50% data availability
- Random drops: Exponential impact on utility with drop probability > 0.3

All results available in `output/data/*.json` and `tee/results/*.json`

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Risk-Hedging Data Trading System            │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   IoT Devices    │      │   Data Seller    │      │   Data Buyer     │
│  (Data Source)   │─────▶│  (Provider)      │◀────▶│  (Consumer)      │
└──────────────────┘      └────────┬─────────┘      └────────┬─────────┘
                                   │                          │
                                   │                          │
                          ┌────────▼──────────────────────────▼─────────┐
                          │    🤖 ML Model Training Pipeline            │
                          │    - LSTM traffic prediction                │
                          │    - Noise robustness testing               │
                          │    - Utility score calculation              │
                          └────────┬────────────────────────────────────┘
                                   │
                                   │ Data + Metadata
                                   │
                          ┌────────▼────────────────────────────────────┐
                          │    🔒 TEE (Intel SGX via Gramine)          │
                          │    - Secure model inference                 │
                          │    - Privacy-preserving computation         │
                          │    - Utility score validation               │
                          │    - ECDSA signature generation             │
                          └────────┬────────────────────────────────────┘
                                   │
                                   │ Signed Utility Score
                                   │
                          ┌────────▼────────────────────────────────────┐
                          │    ⛓️  Blockchain Smart Contract            │
                          │    - Order management                       │
                          │    - Signature verification                 │
                          │    - Atomic payment (P = p + α*k*u)        │
                          │    - Escrow & refund protection             │
                          └─────────────────────────────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │   💰 Settlement   │
                          │   Seller Paid    │
                          └──────────────────┘
```

## 🧩 Key Components Integration

### Data Flow

1. **Data Generation** (ML Component)
   - IoT sensors → PeMS traffic dataset
   - LSTM training & validation
   - Noise injection experiments
   - Output: Utility metrics (R², MAE, RMSE)

2. **TEE Validation** (Security Component)
   - Load trained LSTM model into SGX enclave
   - Secure inference on encrypted data
   - Generate utility score (u = 0.0 to 1.0)
   - Sign with TEE private key → ECDSA signature

3. **Blockchain Settlement** (Smart Contract)
   - Buyer creates order: locks `maxDeposit` ETH
   - Sets pricing: `p_base`, `α`, `k`
   - TEE submits: `(orderId, utility, signature)`
   - Contract verifies signature
   - Calculates: `payment = p_base + α * k * u`
   - Transfers to Seller, refunds excess to Buyer

### Pricing Formula

```
P = p_base + α × k × u

Where:
- p_base: Base fee (guaranteed minimum payment)
- α: Equity share [0, 1] (risk-sharing coefficient)
- k: Utility-to-money conversion factor (scaling parameter)
- u: Data utility score [0, 1] (TEE-verified quality metric)
```

**Example**:
- `p_base = 0.01 ETH`, `α = 0.5`, `k = 0.005 ETH`, `u = 0.95`
- **Payment** = 0.01 + 0.5 × 0.005 × 0.95 = **0.012375 ETH**

## 🛠️ Technology Stack

**Blockchain:**
- Solidity 0.8.28 - Smart contract language
- Hardhat 3.1.0 - Development framework
- Viem 2.41.2 - Type-safe Ethereum library
- OpenZeppelin - Security-audited contracts
- Sepolia/Arbitrum/Optimism - Multi-testnet support

**TEE:**
- Intel SGX - Hardware-based trusted execution
- Gramine 1.7+ - SGX library OS
- Docker - Containerization
- Azure DC-series - SGX-capable VMs

**AI/ML:**
- PyTorch 2.5.1 - Deep learning framework
- CUDA 12.4 - GPU acceleration
- scikit-learn - ML utilities
- NumPy/Pandas - Data processing
- Matplotlib - Visualization

## 🔐 Security Features

1. **TEE Signature Verification**: ECDSA signature validation for utility scores
2. **Escrow Protection**: Funds locked until settlement or timeout refund
3. **Replay Attack Prevention**: Unique nonce per transaction
4. **Reentrancy Guard**: Protection against recursive call attacks
5. **Access Control**: Owner-only admin functions (Ownable pattern)
6. **Atomic Settlement**: All-or-nothing payment execution

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Testnet deployment guide
- [ACCOUNTS.md](ACCOUNTS.md) - Multi-account setup
- [tee/README.md](tee/README.md) - TEE component guide
- [tee/docs/ARCHITECTURE.md](tee/docs/ARCHITECTURE.md) - System architecture
- [tee/docs/SETUP.md](tee/docs/SETUP.md) - SGX setup instructions

## 🚧 Future Enhancements

- [ ] Multi-TEE consensus mechanism
- [ ] Layer 2 deployment (Arbitrum/Optimism mainnet)
- [ ] Real-time IoT device integration
- [ ] Dynamic pricing with market-based α
- [ ] Cross-chain settlement support
- [ ] Advanced data quality metrics

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🎓 Citation

```bibtex
@software{risk_hedging_protocol_2025,
  title={Risk-Hedging Equity Protocol for IoT Data Trading},
  author={Research Team},
  year={2025},
  url={https://github.com/yourusername/Risk-HedgingProtocol},
  note={TEE-secured blockchain data trading with LSTM validation}
}
```

## 🔗 Useful Links

**Testnet Faucets:**
- Sepolia: https://sepoliafaucet.com/
- Arbitrum Sepolia: https://bridge.arbitrum.io/
- Optimism Sepolia: https://app.optimism.io/bridge

**Block Explorers:**
- Sepolia: https://sepolia.etherscan.io/
- Arbitrum: https://sepolia.arbiscan.io/
- Optimism: https://sepolia-optimism.etherscan.io/

**Datasets:**
- PeMS Traffic: http://pems.dot.ca.gov/
- Azure SGX Docs: https://learn.microsoft.com/en-us/azure/virtual-machines/dcv3-series

---

**Project Status**: ✅ Production-ready contracts | 🧪 Research-grade ML/TEE

Built with ❤️ for trustworthy decentralized data trading
