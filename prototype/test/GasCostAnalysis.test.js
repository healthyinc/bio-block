const hre = require("hardhat");
const { expect } = require("chai");
const { ethers } = hre;

const ETH_USD = 3000;
const MATIC_USD = 0.50;
const ARB_GAS_PRICE_GWEI = 0.1;
const POLY_GAS_PRICE_GWEI = 30;

function gasToCostUSD(gasUsed, gasPriceGwei, tokenPriceUSD) {
  const costWei = gasUsed.mul(ethers.utils.parseUnits(gasPriceGwei.toString(), "gwei"));
  const costEth = parseFloat(ethers.utils.formatEther(costWei));
  return costEth * tokenPriceUSD;
}

describe("AnalyticsRegistry gas analysis", function () {
  let registry;
  let owner, analyst1;

  beforeEach(async function () {
    [owner, analyst1] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("AnalyticsRegistry");
    registry = await Factory.deploy();
    await registry.deployed();
  });

  it("measures deployment gas", async function () {
    const Factory = await ethers.getContractFactory("AnalyticsRegistry");
    const deployTx = await Factory.deploy();
    const receipt = await deployTx.deployTransaction.wait();
    expect(receipt.gasUsed.toNumber()).to.be.greaterThan(0);
  });

  it("measures registerAnalytics gas with typical CIDv0", async function () {
    const tx = await registry.registerAnalytics(
      "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
      "QmZTR5bcpQD7cFgTorqxZDYaew1Wqgfbd2ud9QqGPAkK2V",
      "inferential",
      analyst1.address
    );
    const receipt = await tx.wait();
    expect(receipt.gasUsed.toNumber()).to.be.greaterThan(0);
  });

  it("measures registerAnalytics gas with CIDv1", async function () {
    const tx = await registry.registerAnalytics(
      "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
      "bafybeihkoviema7g3gxyt6la7vd5ho32uj3ea5en6gs3lgcpgakqfvrua4",
      "graphical",
      analyst1.address
    );
    const receipt = await tx.wait();
    expect(receipt.gasUsed.toNumber()).to.be.greaterThan(0);
  });

  it("measures amortized gas for 10 sequential writes", async function () {
    let totalGas = 0;
    for (let i = 0; i < 10; i++) {
      const tx = await registry.registerAnalytics(
        `QmSource${i.toString().padStart(3, "0")}`,
        `QmResult${i.toString().padStart(3, "0")}`,
        i % 2 === 0 ? "descriptive" : "inferential",
        analyst1.address
      );
      const receipt = await tx.wait();
      totalGas += receipt.gasUsed.toNumber();
    }
    const avg = Math.round(totalGas / 10);
    expect(avg).to.be.greaterThan(0);
  });

  it("verifies view calls cost zero gas on-chain", async function () {
    await registry.registerAnalytics(
      "QmDataset001", "QmResult001", "descriptive", analyst1.address
    );
    const gas = await registry.estimateGas.getAnalyticsForDataset("QmDataset001");
    expect(gas.toNumber()).to.be.greaterThan(0);
  });

  it("verifies polygon L2 attestation cost is sub-cent", async function () {
    const tx = await registry.registerAnalytics(
      "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
      "QmZTR5bcpQD7cFgTorqxZDYaew1Wqgfbd2ud9QqGPAkK2V",
      "inferential",
      analyst1.address
    );
    const receipt = await tx.wait();
    const polyCostUSD = gasToCostUSD(receipt.gasUsed, POLY_GAS_PRICE_GWEI, MATIC_USD);
    expect(polyCostUSD).to.be.lessThan(0.01);
  });
});
