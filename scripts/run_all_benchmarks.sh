#!/bin/bash

# Run benchmarks on all testnets
# Usage: bash scripts/run_all_benchmarks.sh

echo "=========================================="
echo "Running Benchmarks on All Testnets"
echo "=========================================="
echo ""

# Sepolia
echo "📊 Testing on Sepolia..."
npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
if [ $? -eq 0 ]; then
    echo "✅ Sepolia benchmark completed"
else
    echo "❌ Sepolia benchmark failed"
fi
echo ""

# Arbitrum Sepolia
echo "📊 Testing on Arbitrum Sepolia..."
npx hardhat test test/DataEquityProtocol.benchmark.js --network arbitrumSepolia
if [ $? -eq 0 ]; then
    echo "✅ Arbitrum Sepolia benchmark completed"
else
    echo "❌ Arbitrum Sepolia benchmark failed"
fi
echo ""

# Optimism Sepolia
echo "📊 Testing on Optimism Sepolia..."
npx hardhat test test/DataEquityProtocol.benchmark.js --network optimismSepolia
if [ $? -eq 0 ]; then
    echo "✅ Optimism Sepolia benchmark completed"
else
    echo "❌ Optimism Sepolia benchmark failed"
fi
echo ""

echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved to: output/benchmark/"
echo "=========================================="
