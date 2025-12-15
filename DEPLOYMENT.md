# Complete Deployment Guide

This guide covers deployment of all three components: Smart Contracts, AI/ML Analysis, and TEE Infrastructure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Smart Contract Deployment](#smart-contract-deployment)
3. [ML/AI Setup](#mlai-setup)
4. [TEE Deployment](#tee-deployment)
5. [End-to-End Testing](#end-to-end-testing)

---

## Prerequisites

### Required Tools
- **Node.js 22+** and npm 11+
- **Python 3.9+** with conda
- **Git** for version control
- **MetaMask** or similar wallet
- **Azure CLI** (for TEE deployment)
- **Docker** (for TEE testing)

### Recommended Setup
- 4 Ethereum accounts (Owner, Buyer, Seller, TEE Signer)
- At least 1.5 ETH in testnet tokens across accounts
- Azure subscription with DC-series VM quota (for SGX)

---

## Smart Contract Deployment

### Step 1: Get Testnet ETH

#### Sepolia Faucets
- **Alchemy**: https://sepoliafaucet.com/
- **Infura**: https://www.infura.io/faucet/sepolia
- **QuickNode**: https://faucet.quicknode.com/ethereum/sepolia

#### Arbitrum Sepolia
- **QuickNode**: https://faucet.quicknode.com/arbitrum/sepolia
- **Triangle**: https://faucet.triangleplatform.com/arbitrum/sepolia

#### Optimism Sepolia
- **Optimism**: https://app.optimism.io/faucet
- **QuickNode**: https://faucet.quicknode.com/optimism/sepolia

**Recommended**: 0.5+ ETH per account for comprehensive testing

### Step 2: Configure Multi-Account Setup

See [ACCOUNTS.md](ACCOUNTS.md) for detailed multi-account configuration.

```bash
# Set up 4 accounts (Owner, Buyer, Seller, TEE Signer)
npx hardhat keystore set SEPOLIA_PRIVATE_KEY           # Account 1: Owner
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_2         # Account 2: Buyer
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_3         # Account 3: Seller
npx hardhat keystore set SEPOLIA_PRIVATE_KEY_4         # Account 4: TEE Signer

# Check balances
npx hardhat run scripts/check-all-balances.js --network sepolia
```

## Step 3: Deploy to Testnet

### Deploy to Sepolia (Ethereum L1)

```bash
# Compile contracts
npx hardhat compile

# Deploy
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network sepolia

# Save the deployed address!
# Example output:
# ✔ Confirm deploy to network sepolia (11155111)? … yes
# [ DataEquityProtocolModule ] successfully deployed 🚀
# 
# Deployed Addresses
# DataEquityProtocolModule#DataEquityProtocol - 0x1234...5678
```

### Deploy to Arbitrum Sepolia (L2)

```bash
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network arbitrumSepolia
```

### Deploy to Optimism Sepolia (L2)

```bash
npx hardhat ignition deploy ignition/modules/DataEquityProtocol.ts --network optimismSepolia
```

## Step 4: Verify Contract (Optional)

### Get Etherscan API Key
- Sepolia: https://etherscan.io/myapikey
- Arbitrum: https://arbiscan.io/myapikey
- Optimism: https://optimistic.etherscan.io/myapikey

### Verify Contract

```bash
# Add to .env
ETHERSCAN_API_KEY=your_api_key_here

# Verify on Sepolia
npx hardhat verify --network sepolia <CONTRACT_ADDRESS> <CONSTRUCTOR_ARG>

# Example:
npx hardhat verify --network sepolia 0x1234567890... 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

## Step 5: Run Benchmark Tests

### Basic Test Run

```bash
# On Sepolia
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia

# On Arbitrum Sepolia
npx hardhat test test/DataEquityProtocol.benchmark.js --network arbitrumSepolia

# On Optimism Sepolia
npx hardhat test test/DataEquityProtocol.benchmark.js --network optimismSepolia
```

### Benchmark Metrics

The benchmark test measures:
- **Deployment Cost**: Gas used and ETH cost for contract deployment
- **Order Creation Cost**: Gas and time for creating a data trading order
- **Settlement Cost**: Gas and time for TEE-signed settlement
- **Batch Throughput**: Orders per second for sequential operations
- **Transaction Latency**: End-to-end transaction confirmation time

### Expected Results

**Sepolia (Ethereum L1)**
- Deployment: ~2-3M gas, ~0.01-0.05 ETH
- Order Creation: ~150-200k gas per order
- Settlement: ~100-150k gas
- Confirmation Time: 12-15 seconds

**Arbitrum Sepolia (L2)**
- Deployment: Similar gas, lower ETH cost (10-100x cheaper)
- Order Creation: Similar gas usage
- Settlement: Similar gas usage
- Confirmation Time: 1-2 seconds (faster finality)

**Optimism Sepolia (L2)**
- Deployment: Similar to Arbitrum
- Order Creation: Similar gas usage
- Settlement: Similar gas usage
- Confirmation Time: 1-2 seconds

## Step 6: View Results

Benchmark results are saved to:
```
output/benchmark/benchmark_<network>_<timestamp>.json
```

Example output:
```json
{
  "network": "sepolia",
  "timestamp": "2025-12-15T10:30:00.000Z",
  "tests": [
    {
      "test": "Deployment",
      "gasUsed": "2456789",
      "totalCost": "0.024567 ETH",
      "timeMs": "15234.56"
    },
    {
      "test": "Order Creation",
      "gasUsed": "185432",
      "totalCost": "0.001854 ETH",
      "timeMs": "12456.78"
    }
  ]
}
```

## Step 7: Compare Networks

Run benchmarks on all three networks and compare:

```bash
# Run all benchmarks
./scripts/run_all_benchmarks.sh

# Or manually
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
npx hardhat test test/DataEquityProtocol.benchmark.js --network arbitrumSepolia
npx hardhat test test/DataEquityProtocol.benchmark.js --network optimismSepolia
```

Create comparison script (optional):
```bash
# Create analysis script
python scripts/compare_benchmarks.py output/benchmark/
```

## Troubleshooting

### Error: "insufficient funds"
- Get more testnet ETH from faucets
- Check balance: `npx hardhat run scripts/check-balance.js --network sepolia`

### Error: "nonce too low"
- Reset account nonce in MetaMask: Settings → Advanced → Reset Account

### Error: "transaction underpriced"
- Increase gas price in hardhat.config.ts:
```typescript
sepolia: {
  type: "http",
  url: "...",
  accounts: ["..."],
  gasPrice: 20000000000, // 20 gwei
}
```

### Slow confirmation
- Normal on Sepolia (12-15s per block)
- Consider using L2s (Arbitrum/Optimism) for faster finality

## Cost Estimation

**Full deployment + 10 orders + 10 settlements:**
- Sepolia: ~0.1-0.2 ETH
- Arbitrum Sepolia: ~0.001-0.01 ETH (100x cheaper)
- Optimism Sepolia: ~0.001-0.01 ETH (100x cheaper)

## Block Explorers

- **Sepolia**: https://sepolia.etherscan.io/
- **Arbitrum Sepolia**: https://sepolia.arbiscan.io/
- **Optimism Sepolia**: https://sepolia-optimism.etherscan.io/

Search for your transaction hash or contract address to view details.

## Next Steps

1. ✅ Deploy to testnet
2. ✅ Run benchmark tests
3. 📊 Analyze gas costs and performance
4. 📝 Document results in paper/README
5. 🚀 Consider mainnet deployment (after thorough testing and audit)
