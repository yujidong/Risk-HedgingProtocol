import { network } from "hardhat";
import { formatEther } from "viem";

async function main() {
  const { viem } = await network.connect();
  const [deployer] = await viem.getWalletClients();
  const publicClient = await viem.getPublicClient();

  console.log("\n=== Account Information ===");
  console.log("Network:", network.name);
  console.log("Address:", deployer.account.address);
  
  const balance = await publicClient.getBalance({ 
    address: deployer.account.address 
  });
  
  console.log("Balance:", formatEther(balance), "ETH");
  
  const blockNumber = await publicClient.getBlockNumber();
  console.log("Current Block:", blockNumber.toString());
  
  // Check if sufficient for deployment
  const estimatedDeploymentCost = 0.05; // ETH
  const balanceInEth = parseFloat(formatEther(balance));
  
  if (balanceInEth < estimatedDeploymentCost) {
    console.log("\n⚠️  WARNING: Low balance!");
    console.log(`   Need ~${estimatedDeploymentCost} ETH for deployment`);
    console.log("   Get testnet ETH from faucets (see DEPLOYMENT.md)");
  } else {
    console.log("\n✅ Sufficient balance for deployment");
  }
  
  console.log("=".repeat(50) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
