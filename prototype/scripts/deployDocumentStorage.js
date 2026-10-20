const hre = require("hardhat");

async function main() {
  const DocumentStorage = await hre.ethers.getContractFactory("DocumentStorage");
  const storage = await DocumentStorage.deploy();
  await storage.deployed();
  console.log("DocumentStorage deployed to:", storage.address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
