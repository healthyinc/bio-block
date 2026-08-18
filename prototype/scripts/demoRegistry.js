const hre = require("hardhat");

async function main() {
  console.log("🚀 Starting AnalyticsRegistry Demo...\n");

  // 1. Get signers (simulating different users)
  const [deployer, analyst1, analyst2] = await hre.ethers.getSigners();
  console.log(`👤 Deployer address: ${deployer.address}`);
  console.log(`👤 Analyst 1 address: ${analyst1.address}`);
  console.log(`👤 Analyst 2 address: ${analyst2.address}\n`);

  // 2. Deploy the contract
  console.log("⏳ Deploying AnalyticsRegistry contract...");
  const AnalyticsRegistry = await hre.ethers.getContractFactory("AnalyticsRegistry");
  const registry = await AnalyticsRegistry.deploy();
  await registry.deployed();
  console.log(`✅ Contract deployed to: ${registry.address}\n`);

  // 3. Register some mock analytics
  console.log("📝 Analyst 1 is registering a descriptive analysis...");
  const sourceCID = "QmOriginalDataset123456789";
  let tx = await registry.connect(analyst1).registerAnalytics(
    sourceCID,
    "QmDescriptiveResultABC",
    "descriptive"
  );
  await tx.wait();
  console.log("✅ Successfully registered descriptive analysis!\n");

  console.log("📝 Analyst 2 is registering a visualization for the SAME dataset...");
  tx = await registry.connect(analyst2).registerAnalytics(
    sourceCID,
    "QmVisualizationResultXYZ",
    "graphical"
  );
  await tx.wait();
  console.log("✅ Successfully registered visualization!\n");

  // 4. Fetch the data back!
  console.log("🔍 Fetching all analytics linked to the original dataset...");
  const resultsForDataset = await registry.getAnalyticsForDataset(sourceCID);
  console.log(`📊 Found ${resultsForDataset.length} results for dataset ${sourceCID}:`);
  resultsForDataset.forEach((cid, index) => {
    console.log(`   ${index + 1}. Result CID: ${cid}`);
  });
  console.log("\n");

  console.log("🔍 Fetching Analyst 1's personal analytics history...");
  const analyst1History = await registry.connect(analyst1).getMyAnalytics(0, 10);
  console.log(`📊 Analyst 1 has ${analyst1History.length} records in their history:`);
  analyst1History.forEach((record, index) => {
    console.log(`   ${index + 1}. Type: ${record.analysisType} | Source: ${record.sourceCID} -> Result: ${record.resultCID}`);
  });
  
  console.log("\n🎉 Demo completed successfully!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
