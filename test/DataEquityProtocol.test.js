import { test } from "node:test";
import { strict as assert } from "node:assert";
import { network } from "hardhat";
import { parseEther, parseUnits, keccak256, toBytes, encodePacked } from "viem";

test("DataEquityProtocol - Basic Functionality", async (t) => {
  const { viem } = await network.connect();
  const SCALE_FACTOR = parseUnits("1", 18);
  
  await t.test("Deployment", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    const teeSignerAddress = await contract.read.teeSigner();
    assert.equal(teeSignerAddress.toLowerCase(), teeSigner.account.address.toLowerCase(), "TEE signer should match");
    
    const ownerAddress = await contract.read.owner();
    assert.equal(ownerAddress.toLowerCase(), owner.account.address.toLowerCase(), "Owner should match");
    
    console.log("✓ Contract deployed successfully");
    console.log("  Contract address:", contract.address);
    console.log("  TEE Signer:", teeSignerAddress);
    console.log("  Owner:", ownerAddress);
  });

  await t.test("Order Creation", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n; // 0.5
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.1");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("test-nonce-1"));

    const hash = await contract.write.createOrder(
      [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: buyer.account }
    );

    await publicClient.waitForTransactionReceipt({ hash });
    
    const order = await contract.read.orders([1n]);
    assert.equal(order[1].toLowerCase(), buyer.account.address.toLowerCase(), "Buyer should match");
    assert.equal(order[2].toLowerCase(), seller.account.address.toLowerCase(), "Seller should match");
    assert.equal(order[3], p_base, "Base price should match");
    
    console.log("✓ Order created successfully");
    console.log("  Order ID: 1");
    console.log("  Buyer:", order[1]);
    console.log("  Seller:", order[2]);
    console.log("  Base Price:", order[3].toString());
  });

  await t.test("Order Settlement", async () => {
    const [owner, buyer, seller, teeSigner] = await viem.getWalletClients();
    const publicClient = await viem.getPublicClient();
    
    const contract = await viem.deployContract("DataEquityProtocol", [teeSigner.account.address]);
    
    const p_base = parseEther("0.01");
    const alpha = SCALE_FACTOR / 2n;
    const k_factor = parseEther("0.005");
    const maxDeposit = parseEther("0.1");
    const durationSeconds = 3600n;
    const nonce = keccak256(toBytes("test-nonce-settle"));

    const createHash = await contract.write.createOrder(
      [seller.account.address, p_base, alpha, k_factor, nonce, durationSeconds],
      { value: maxDeposit, account: buyer.account }
    );
    
    await publicClient.waitForTransactionReceipt({ hash: createHash });
    
    const orderId = 1n;
    const utility = SCALE_FACTOR * 95n / 100n; // 0.95 utility
    
    const messageHash = keccak256(
      encodePacked(
        ["uint256", "uint256", "bytes32"],
        [orderId, utility, nonce]
      )
    );
    
    const signature = await teeSigner.signMessage({ 
      message: { raw: messageHash }
    });

    const sellerBalanceBefore = await publicClient.getBalance({ address: seller.account.address });
    
    const settleHash = await contract.write.settleTransaction(
      [orderId, utility, signature],
      { account: buyer.account }
    );
    
    await publicClient.waitForTransactionReceipt({ hash: settleHash });
    
    const sellerBalanceAfter = await publicClient.getBalance({ address: seller.account.address });
    assert.ok(sellerBalanceAfter > sellerBalanceBefore, "Seller should receive payment");
    
    const order = await contract.read.orders([orderId]);
    assert.equal(order[9], 2, "Order should be in SETTLED state");
    
    console.log("✓ Order settled successfully");
    console.log("  Utility:", utility.toString());
    console.log("  Seller received payment");
    console.log("  Order state: SETTLED");
  });
});
