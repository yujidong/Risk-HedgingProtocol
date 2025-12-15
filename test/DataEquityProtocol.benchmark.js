import { test } from "node:test";
import { strict as assert } from "node:assert";
import { network } from "hardhat";
import { parseEther, parseUnits, keccak256, toBytes, encodePacked, formatEther } from "viem";
import { performance } from "perf_hooks";

/**
 * Benchmark test suite for DataEquityProtocol on public testnets
 * Measures gas costs, transaction times, and throughput
 * 
 * Usage:
 * npx hardhat test test/DataEquityProtocol.benchmark.js --network sepolia
 * npx hardhat test test/DataEquityProtocol.benchmark.js --network arbitrumSepolia
 * npx hardhat test test/DataEquityProtocol.benchmark.js --network optimismSepolia
 */

test("DataEquityProtocol - Benchmark Tests", async (t) => {
  const { viem } = await network.connect();
  const SCALE_FACTOR = parseUnits("1", 18);
  
  let benchmarkResults = {
    network: network.name,
    timestamp: new Date().toISOString(),
    tests: []
  };

  // Pre-check: Estimate required balance for all tests
  await t.test("Balance Pre-Check", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    console.log("\n" + "=".repeat(60));
    console.log("BALANCE PRE-CHECK & TEST ESTIMATION");
    console.log("=".repeat(60));
    
    const ownerBalance = await publicClient.getBalance({ address: owner.account.address });
    const buyerBalance = await publicClient.getBalance({ address: buyer.account.address });
    const sellerBalance = await publicClient.getBalance({ address: seller.account.address });
    
    console.log("\nCurrent Balances:");
    console.log("  Owner:", formatEther(ownerBalance), "ETH");
    console.log("  Buyer:", formatEther(buyerBalance), "ETH");
    console.log("  Seller:", formatEther(sellerBalance), "ETH");
    
    // Estimate required amounts
    const depositPerOrder = parseEther("0.02");
    const estimatedGasPerTx = parseEther("0.001"); // Conservative estimate
    
    const deploymentCost = estimatedGasPerTx;
    const orderCreationCost = depositPerOrder + estimatedGasPerTx;
    const settlementCost = depositPerOrder + estimatedGasPerTx * 2n; // create + settle
    const batchOperationsCost = depositPerOrder * 3n + estimatedGasPerTx * 3n;
    const refundCost = depositPerOrder + estimatedGasPerTx * 2n; // Will be refunded
    const edgeCasesCost = depositPerOrder * 3n + estimatedGasPerTx * 6n; // 3 orders, 6 txs
    const batchSettlementCost = depositPerOrder * 3n + estimatedGasPerTx * 6n; // 3 create + 3 settle
    
    const totalOwnerRequired = deploymentCost * 7n; // 7 contract deployments
    const totalBuyerRequired = orderCreationCost + settlementCost + batchOperationsCost + 
                                refundCost + edgeCasesCost + batchSettlementCost;
    
    console.log("\nEstimated Requirements:");
    console.log("  Owner needs:", formatEther(totalOwnerRequired), "ETH (deployments)");
    console.log("  Buyer needs:", formatEther(totalBuyerRequired), "ETH (transactions)");
    console.log("\nTest Breakdown:");
    console.log("  - Order Creation: ", formatEther(orderCreationCost), "ETH");
    console.log("  - Settlement: ", formatEther(settlementCost), "ETH");
    console.log("  - Batch Ops (3x): ", formatEther(batchOperationsCost), "ETH");
    console.log("  - Refund (returned): ", formatEther(refundCost), "ETH");
    console.log("  - Edge Cases (3x): ", formatEther(edgeCasesCost), "ETH");
    console.log("  - Batch Settlement (3x): ", formatEther(batchSettlementCost), "ETH");
    
    const ownerSufficient = ownerBalance >= totalOwnerRequired;
    const buyerSufficient = buyerBalance >= totalBuyerRequired;
    
    console.log("\nBalance Check:");
    console.log("  Owner:", ownerSufficient ? "✅ Sufficient" : "❌ Insufficient");
    console.log("  Buyer:", buyerSufficient ? "✅ Sufficient" : "❌ Insufficient");
    
    if (!ownerSufficient) {
      const needed = totalOwnerRequired - ownerBalance;
      console.log("\n⚠️  Owner needs", formatEther(needed), "more ETH");
      console.log("   Get testnet ETH from: https://sepoliafaucet.com/");
    }
    
    if (!buyerSufficient) {
      const needed = totalBuyerRequired - buyerBalance;
      console.log("\n⚠️  Buyer needs", formatEther(needed), "more ETH");
      console.log("   Get testnet ETH from: https://sepoliafaucet.com/");
    }
    
    console.log("\n" + "=".repeat(60));
    
    if (!ownerSufficient || !buyerSufficient) {
      throw new Error("Insufficient balance for complete benchmark run. Please add testnet ETH.");
    }
    
    console.log("✅ All balances sufficient. Proceeding with benchmark tests...\n");
  });

  await t.test("Deployment Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    console.log("\n=== Deployment Benchmark ===");
    console.log("Network:", network.name);
    console.log("Deployer:", owner.account.address);
    
    const balance = await publicClient.getBalance({ address: owner.account.address });
    console.log("Balance:", formatEther(balance), "ETH");
    
    const startTime = performance.now();
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
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
    
    console.log("\n📊 Deployment Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", deployTime.toFixed(2), "ms");
    console.log("  Contract:", contract.address);
    console.log("  Tx Hash:", deployHash);
    
    assert.ok(contract.address, "Contract should be deployed");
  });

  await t.test("Order Creation Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Order Creation Benchmark ===");
    
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.02");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("benchmark-order-1"));

    const startTime = performance.now();
    const hash = await contract.write.createOrder(
      [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: buyer.account }
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
    
    console.log("\n📊 Order Creation Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", createTime.toFixed(2), "ms");
    console.log("  Tx Hash:", hash);
    
    assert.equal(receipt.status, "success", "Transaction should succeed");
  });

  await t.test("Settlement Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Settlement Benchmark ===");
    
    // Create order first
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.02");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("benchmark-settle-1"));

    const createHash = await contract.write.createOrder(
      [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: buyer.account }
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
    
    const signature = await teeSigner.signMessage({ 
      message: { raw: messageHash }
    });

    const startTime = performance.now();
    const settleHash = await contract.write.settleTransaction(
      [orderId, utility, signature],
      { account: buyer.account }
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
    
    console.log("\n📊 Settlement Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", settleTime.toFixed(2), "ms");
    console.log("  Tx Hash:", settleHash);
    
    assert.equal(receipt.status, "success", "Settlement should succeed");
  });

  await t.test("Batch Operations Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Batch Operations Benchmark ===");
    
    const batchSize = 3;
    const results = [];
    
    console.log(`Creating ${batchSize} orders sequentially...`);
    
    const batchStartTime = performance.now();
    
    for (let i = 0; i < batchSize; i++) {
      const p_base = parseEther("0.01");
      const alpha = SCALE_FACTOR / 2n;
      const k_factor = parseEther("0.005");
      const maxDeposit = parseEther("0.1");
      const durationSeconds = 3600n;
      const nonce = keccak256(toBytes(`batch-order-${i}`));

      const startTime = performance.now();
      const hash = await contract.write.createOrder(
        [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
        { value: maxDeposit, account: buyer.account }
      );
      
      const receipt = await publicClient.waitForTransactionReceipt({ hash });
      const orderTime = performance.now() - startTime;
      
      results.push({
        orderNum: i + 1,
        gasUsed: receipt.gasUsed,
        timeMs: orderTime
      });
      
      console.log(`  Order ${i + 1}: ${receipt.gasUsed.toString()} gas, ${orderTime.toFixed(2)}ms`);
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
      throughputTxPerSec: (batchSize / (totalBatchTime / 1000)).toFixed(2)
    };
    
    benchmarkResults.tests.push(batchResult);
    
    console.log("\n📊 Batch Results:");
    console.log("  Total Time:", totalBatchTime.toFixed(2), "ms");
    console.log("  Average Gas:", avgGasUsed.toString());
    console.log("  Average Time:", avgTime.toFixed(2), "ms");
    console.log("  Throughput:", batchResult.throughputTxPerSec, "tx/sec");
    
    assert.equal(results.length, batchSize, "All orders should be created");
  });

  await t.test("Refund Mechanism Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Refund Mechanism Benchmark ===");
    
    // Create order with immediate deadline (already expired at creation time)
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.02");
    const durationSeconds = 1n; // 1 second duration
    const nonce = keccak256(toBytes("refund-test-1"));

    const createHash = await contract.write.createOrder(
      [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: buyer.account }
    );
    
    await publicClient.waitForTransactionReceipt({ hash: createHash });
    console.log("  Order created, waiting for deadline to pass...");
    
    // Mine a few blocks to ensure deadline passes (Sepolia block time ~12s)
    // Send a dummy transaction to trigger new block
    const dummyHash = await buyer.sendTransaction({
      to: buyer.account.address,
      value: 0n
    });
    await publicClient.waitForTransactionReceipt({ hash: dummyHash });
    console.log("  Deadline should now be passed");
    
    const orderId = 1n;
    const startTime = performance.now();
    const refundHash = await contract.write.refundOrder(
      [orderId],
      { account: buyer.account }
    );
    const refundTime = performance.now() - startTime;
    
    const receipt = await publicClient.waitForTransactionReceipt({ hash: refundHash });
    
    const result = {
      test: "Refund",
      gasUsed: receipt.gasUsed.toString(),
      gasPrice: receipt.effectiveGasPrice.toString(),
      totalCost: formatEther(receipt.gasUsed * receipt.effectiveGasPrice),
      timeMs: refundTime.toFixed(2),
      txHash: refundHash
    };
    
    benchmarkResults.tests.push(result);
    
    console.log("\n📊 Refund Results:");
    console.log("  Gas Used:", receipt.gasUsed.toString());
    console.log("  Gas Price:", formatEther(receipt.effectiveGasPrice), "ETH/gas");
    console.log("  Total Cost:", result.totalCost, "ETH");
    console.log("  Time:", refundTime.toFixed(2), "ms");
    console.log("  Tx Hash:", refundHash);
    
    assert.equal(receipt.status, "success", "Refund should succeed");
  });

  await t.test("Edge Cases Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Edge Cases Benchmark ===");
    
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.02");
    const durationSeconds = 3600n;
    
    const edgeCases = [
      { name: "Utility 0%", utility: 0n },
      { name: "Utility 50%", utility: SCALE_FACTOR / 2n },
      { name: "Utility 100%", utility: SCALE_FACTOR }
    ];
    
    for (const edgeCase of edgeCases) {
      const nonce = keccak256(toBytes(`edge-${edgeCase.name}`));
      
      // Create order
      const createHash = await contract.write.createOrder(
        [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
        { value: maxDeposit, account: buyer.account }
      );
      await publicClient.waitForTransactionReceipt({ hash: createHash });
      
      // Get order ID
      const orderId = await contract.read.orderCounter();
      
      // Sign with TEE
      const messageHash = keccak256(
        encodePacked(
          ["uint256", "uint256", "bytes32"],
          [orderId, edgeCase.utility, nonce]
        )
      );
      const signature = await teeSigner.signMessage({ message: { raw: messageHash } });
      
      // Settle
      const startTime = performance.now();
      const settleHash = await contract.write.settleTransaction(
        [orderId, edgeCase.utility, signature],
        { account: buyer.account }
      );
      const settleTime = performance.now() - startTime;
      
      const receipt = await publicClient.waitForTransactionReceipt({ hash: settleHash });
      
      const result = {
        test: `Edge Case - ${edgeCase.name}`,
        utility: edgeCase.utility.toString(),
        gasUsed: receipt.gasUsed.toString(),
        gasPrice: receipt.effectiveGasPrice.toString(),
        totalCost: formatEther(receipt.gasUsed * receipt.effectiveGasPrice),
        timeMs: settleTime.toFixed(2)
      };
      
      benchmarkResults.tests.push(result);
      
      console.log(`\n  ${edgeCase.name}:`);
      console.log(`    Gas Used: ${receipt.gasUsed.toString()}`);
      console.log(`    Time: ${settleTime.toFixed(2)}ms`);
    }
    
    console.log("\n✓ All edge cases tested");
  });

  await t.test("Batch Settlement Benchmark", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    console.log("\n=== Batch Settlement Benchmark ===");
    
    const batchSize = 3;
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.02");
    const durationSeconds = 3600n;
    
    // Create multiple orders
    console.log(`Creating ${batchSize} orders...`);
    const orderIds = [];
    const nonces = [];
    
    for (let i = 0; i < batchSize; i++) {
      const nonce = keccak256(toBytes(`batch-settle-${i}`));
      nonces.push(nonce);
      
      const hash = await contract.write.createOrder(
        [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
        { value: maxDeposit, account: buyer.account }
      );
      await publicClient.waitForTransactionReceipt({ hash });
      
      const orderId = await contract.read.orderCounter();
      orderIds.push(orderId);
    }
    
    console.log(`Settling ${batchSize} orders...`);
    const batchStartTime = performance.now();
    const results = [];
    
    for (let i = 0; i < batchSize; i++) {
      const utility = SCALE_FACTOR * 80n / 100n; // 0.8 utility
      const messageHash = keccak256(
        encodePacked(
          ["uint256", "uint256", "bytes32"],
          [orderIds[i], utility, nonces[i]]
        )
      );
      const signature = await teeSigner.signMessage({ message: { raw: messageHash } });
      
      const startTime = performance.now();
      const hash = await contract.write.settleTransaction(
        [orderIds[i], utility, signature],
        { account: buyer.account }
      );
      const receipt = await publicClient.waitForTransactionReceipt({ hash });
      const settleTime = performance.now() - startTime;
      
      results.push({
        orderNum: i + 1,
        gasUsed: receipt.gasUsed,
        timeMs: settleTime
      });
      
      console.log(`  Settlement ${i + 1}: ${receipt.gasUsed.toString()} gas, ${settleTime.toFixed(2)}ms`);
    }
    
    const totalBatchTime = performance.now() - batchStartTime;
    const avgGasUsed = results.reduce((sum, r) => sum + r.gasUsed, 0n) / BigInt(batchSize);
    const avgTime = results.reduce((sum, r) => sum + r.timeMs, 0) / batchSize;
    
    const batchResult = {
      test: "Batch Settlement",
      batchSize: batchSize,
      totalTimeMs: totalBatchTime.toFixed(2),
      avgGasUsed: avgGasUsed.toString(),
      avgTimeMs: avgTime.toFixed(2),
      throughputTxPerSec: (batchSize / (totalBatchTime / 1000)).toFixed(2)
    };
    
    benchmarkResults.tests.push(batchResult);
    
    console.log("\n📊 Batch Settlement Results:");
    console.log("  Total Time:", totalBatchTime.toFixed(2), "ms");
    console.log("  Average Gas:", avgGasUsed.toString());
    console.log("  Average Time:", avgTime.toFixed(2), "ms");
    console.log("  Throughput:", batchResult.throughputTxPerSec, "tx/sec");
    
    assert.equal(results.length, batchSize, "All settlements should complete");
  });

  // Save benchmark results
  await t.test("Save Results", async () => {
    const fs = await import("fs");
    const path = await import("path");
    
    const outputDir = path.join(process.cwd(), "output", "benchmark");
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const filename = `benchmark_${network.name}_${Date.now()}.json`;
    const filepath = path.join(outputDir, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(benchmarkResults, null, 2));
    
    console.log("\n\n" + "=".repeat(60));
    console.log("📈 BENCHMARK SUMMARY");
    console.log("=".repeat(60));
    console.log("Network:", network.name);
    console.log("Total Tests:", benchmarkResults.tests.length);
    console.log("Results saved to:", filepath);
    console.log("=".repeat(60) + "\n");
  });
});
