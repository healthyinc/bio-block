const hre = require("hardhat");

async function main() {
  console.log("Deploying AnalyticsRegistry to Sepolia...\n");

  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer address: ${deployer.address}`);

  const balance = await deployer.getBalance();
  console.log(`Deployer balance: ${hre.ethers.utils.formatEther(balance)} ETH\n`);

  if (balance.eq(0)) {
    console.error(
      "Error: Deployer has 0 ETH. Fund with Sepolia ETH from https://sepoliafaucet.com"
    );
    process.exit(1);
  }

  console.log("Deploying contract...");
  const AnalyticsRegistry = await hre.ethers.getContractFactory("AnalyticsRegistry");
  const registry = await AnalyticsRegistry.deploy();
  await registry.deployed();

  console.log(`\nAnalyticsRegistry deployed to: ${registry.address}`);
  console.log(`Etherscan: https://sepolia.etherscan.io/address/${registry.address}`);
  console.log(`\nSave this address for your analytics-service .env:`);
  console.log(`   ANALYTICS_REGISTRY_ADDRESS=${registry.address}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Deployment failed:", error);
    process.exit(1);
  });
