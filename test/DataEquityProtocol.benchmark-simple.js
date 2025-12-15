import { test } from "node:test";
import { strict as assert } from "node:assert";
import { network } from "hardhat";
import { parseEther, parseUnits, keccak256, toBytes, encodePacked, formatEther } from "viem";
import { performance } from "perf_hooks";

/**
 * Simplified Benchmark for Testnet Deployment
 * Uses single account for buyer/seller/TEE roles
 * 
 * Usage: npx hardhat test test/DataEquityProtocol.benchmark-simple.js --network sepolia
 */

test("DataEquityProtocol - Testnet Benchmark", async (t) => {
  const { viem } = await network.connect();
  const SCALE_FACTOR = parseUnits("1", 18);
  
  let benchmarkResults = {
    network: network.name || "testnet",
    timestamp: new Date().toISOString(),
    tests: []
  };

  await t.test("Deployment Benchmark", async () => {
    const [deployer] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    console.log("\n" + "=".repeat(60));
    console.log("DEPLOYMENT BENCHMARK");
    console.log("=".repeat(60));
    console.log("Network:", network.name || "testnet");
    console.log("Deployer:", deployer.account.address);
    
    const balance = await publicClient.getBalance({ address: deployer.account.address });
    console.log("Balance:", formatEther(balance), "ETH");
    
    const startTime = performance.now();
    const contract = await viem.deployContract("DataEquityProtocol", [deployer.account.address]);
    const deployTime = performance.now() - startTime;
    
    // Get deployment receipt from recent block
    const latestBlock = await publicClient.getBlockNumber();
    const block = await publicClient.getBlock({ blockNumber: latestBlock });
    const deployHash = block.transactions[block.transactions.length - 1];
    const receipt = await publicClient.getTransactionReceipt({ hash: deployHash });
    
    const result = {
      test: "Deployment",
      gasUsed: receipt.gasUsed.toString(),
      gasPrice: receipt.effectiveGasPrice.toString(),
      totalCost: formatEther(receipt.gasUsed * receipt.effectiveGasPrice),
      timeMs: deployTime.toFixed(2),
      txHash: deployHash,
      contractAddress: contract.address
    };
    
    benchmarkResults.tests.push(result);
    
    console.log("\n📊 Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", deployTime.toFixed(2), "ms");
    console.log("  Contract:", contract.address);
    console.log("  Tx Hash:", deployHash);
    console.log("  Explorer:", getExplorerUrl(network.name, deployHash));
    
    assert.ok(contract.address, "Contract should be deployed");
  });

  await t.test("Order Creation Benchmark", async () => {
    const [account] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    // Deploy contract
    const contract = await viem.deployContract("DataEquityProtocol", [account.account.address]);
    
    console.log("\n" + "=".repeat(60));
    console.log("ORDER CREATION BENCHMARK");
    console.log("=".repeat(60));
    
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.05");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("benchmark-order-1"));

    const startTime = performance.now();
    const hash = await contract.write.createOrder(
      [account.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: account.account }
    );
    const createTime = performance.now() - startTime;

    const receipt = await publicClient.waitForTransactionReceipt({ hash });
    
    const result = {
      test: "Order Creation",
      gasUsed: receipt.gasUsed.toString(),
      gasPrice: receipt.effectiveGasPrice.toString(),
      totalCost: formatEther(receipt.gasUsed * receipt.effectiveGasPrice),
      timeMs: createTime.toFixed(2),
      txHash: hash
    };
    
    benchmarkResults.tests.push(result);
    
    console.log("\n📊 Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", createTime.toFixed(2), "ms");
    console.log("  Tx Hash:", hash);
    console.log("  Explorer:", getExplorerUrl(network.name, hash));
    
    assert.equal(receipt.status, "success", "Transaction should succeed");
  });

  await t.test("Settlement Benchmark", async () => {
    const [account] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [account.account.address]);
    
    console.log("\n" + "=".repeat(60));
    console.log("SETTLEMENT BENCHMARK");
    console.log("=".repeat(60));
    
    // Create order first
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.05");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("benchmark-settle-1"));

    const createHash = await contract.write.createOrder(
      [account.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: account.account }
    );
    
    await publicClient.waitForTransactionReceipt({ hash: createHash });
    
    // Prepare settlement
    const orderId = 1n;
    const utility = SCALE_FACTOR * 95n / 100n;
    
    const messageHash = keccak256(
      encodePacked(
        ["uint256", "uint256", "bytes32"],
        [orderId, utility, nonce]
      )
    );
    
    const signature = await account.signMessage({ 
      message: { raw: messageHash }
    });

    const startTime = performance.now();
    const settleHash = await contract.write.settleTransaction(
      [orderId, utility, signature],
      { account: account.account }
    );
    const settleTime = performance.now() - startTime;
    
    const receipt = await publicClient.waitForTransactionReceipt({ hash: settleHash });
    
    const result = {
      test: "Settlement",
      gasUsed: receipt.gasUsed.toString(),
      gasPrice: receipt.effectiveGasPrice.toString(),
      totalCost: formatEther(receipt.gasUsed * receipt.effectiveGasPrice),
      timeMs: settleTime.toFixed(2),
      txHash: settleHash
    };
    
    benchmarkResults.tests.push(result);
    
    console.log("\n📊 Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", settleTime.toFixed(2), "ms");
    console.log("  Tx Hash:", settleHash);
    console.log("  Explorer:", getExplorerUrl(network.name, settleHash));
    
    assert.equal(receipt.status, "success", "Settlement should succeed");
  });

  await t.test("Batch Operations Benchmark", async () => {
    const [account] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [account.account.address]);
    
    console.log("\n" + "=".repeat(60));
    console.log("BATCH OPERATIONS BENCHMARK");
    console.log("=".repeat(60));
    
    const batchSize = 3; // Reduced for testnet
    const results = [];
    
    console.log(`Creating ${batchSize} orders sequentially...\n`);
    
    const batchStartTime = performance.now();
    
    for (let i = 0; i < batchSize; i++) {
      const p_base = parseEther("0.01");
      const alpha = SCALE_FACTOR / 2n;
      const k_factor = parseEther("0.005");
      const maxDeposit = parseEther("0.05");
      const durationSeconds = 3600n;
      const nonce = keccak256(toBytes(`batch-order-${i}`));

      const startTime = performance.now();
      const hash = await contract.write.createOrder(
        [account.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
        { value: maxDeposit, account: account.account }
      );
      
      const receipt = await publicClient.waitForTransactionReceipt({ hash });
      const orderTime = performance.now() - startTime;
      
      results.push({
        orderNum: i + 1,
        gasUsed: receipt.gasUsed,
        timeMs: orderTime,
        txHash: hash
      });
      
      console.log(`  Order ${i + 1}:`);
      console.log(`    Gas: ${receipt.gasUsed.toString()}`);
      console.log(`    Time: ${orderTime.toFixed(2)}ms`);
      console.log(`    Tx: ${hash}`);
    }
    
    const totalBatchTime = performance.now() - batchStartTime;
    const avgGasUsed = results.reduce((sum, r) => sum + r.gasUsed, 0n) / BigInt(batchSize);
    const avgTime = results.reduce((sum, r) => sum + r.timeMs, 0) / batchSize;
    
    const batchResult = {
      test: "Batch Operations",
      batchSize: batchSize,
      totalTimeMs: totalBatchTime.toFixed(2),
      avgGasUsed: avgGasUsed.toString(),
      avgTimeMs: avgTime.toFixed(2),
      throughputTxPerSec: (batchSize / (totalBatchTime / 1000)).toFixed(4)
    };
    
    benchmarkResults.tests.push(batchResult);
    
    console.log("\n📊 Batch Results:");
    console.log("  Total Time:", totalBatchTime.toFixed(2), "ms");
    console.log("  Average Gas:", avgGasUsed.toString());
    console.log("  Average Time:", avgTime.toFixed(2), "ms/tx");
    console.log("  Throughput:", batchResult.throughputTxPerSec, "tx/sec");
    
    assert.equal(results.length, batchSize, "All orders should be created");
  });

  // Save results
  await t.test("Save Results", async () => {
    const fs = await import("fs");
    const path = await import("path");
    
    const outputDir = path.join(process.cwd(), "output", "benchmark");
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const networkName = network.name || "testnet";
    const filename = `benchmark_${networkName}_${Date.now()}.json`;
    const filepath = path.join(outputDir, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(benchmarkResults, null, 2));
    
    console.log("\n\n" + "=".repeat(60));
    console.log("📈 BENCHMARK SUMMARY");
    console.log("=".repeat(60));
    console.log("Network:", networkName);
    console.log("Total Tests:", benchmarkResults.tests.length);
    
    // Calculate total costs
    let totalCost = 0;
    for (const test of benchmarkResults.tests) {
      if (test.totalCost) {
        totalCost += parseFloat(test.totalCost);
      }
    }
    
    console.log("Total Cost:", totalCost.toFixed(6), "ETH");
    console.log("\nResults saved to:", filename);
    console.log("=".repeat(60) + "\n");
  });
});

function getExplorerUrl(networkName, txHash) {
  const explorers = {
    sepolia: "https://sepolia.etherscan.io/tx/",
    arbitrumSepolia: "https://sepolia.arbiscan.io/tx/",
    optimismSepolia: "https://sepolia-optimism.etherscan.io/tx/"
  };
  
  const baseUrl = explorers[networkName] || "https://etherscan.io/tx/";
  return baseUrl + txHash;
}
