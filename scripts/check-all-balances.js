import { network } from "hardhat";
import { formatEther } from "viem";

async function main() {
  const { viem } = await network.connect();
  const walletClients = await viem.getWalletClients();
  const publicClient = await viem.getPublicClient();

  console.log("\n=== Multi-Account Information ===");
  console.log("Network:", network.name);
  console.log("Total Accounts:", walletClients.length);
  console.log("");
  
  let totalBalance = 0n;
  
  for (let i = 0; i < walletClients.length; i++) {
    const wallet = walletClients[i];
    const balance = await publicClient.getBalance({ 
      address: wallet.account.address 
    });
    
    totalBalance += balance;
    
    console.log(`Account ${i + 1}:`);
    console.log(`  Address: ${wallet.account.address}`);
    console.log(`  Balance: ${formatEther(balance)} ETH`);
    console.log("");
  }
  
  console.log("Total Balance:", formatEther(totalBalance), "ETH");
  
  const blockNumber = await publicClient.getBlockNumber();
  console.log("Current Block:", blockNumber.toString());
  
  // Check if sufficient for testing
  const estimatedCost = 0.1; // ETH for comprehensive testing
  const totalInEth = parseFloat(formatEther(totalBalance));
  
  console.log("\n" + "=".repeat(50));
  if (totalInEth < estimatedCost) {
    console.log("⚠️  WARNING: Low total balance!");
    console.log(`   Recommended: ${estimatedCost} ETH for full testing`);
    console.log("   Get testnet ETH from faucets (see DEPLOYMENT.md)");
  } else {
    console.log("✅ Sufficient balance for comprehensive testing");
  }
  console.log("=".repeat(50) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
