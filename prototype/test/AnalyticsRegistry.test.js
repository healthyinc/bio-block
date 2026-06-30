const hre = require("hardhat");
const { expect } = require("chai");
const { ethers } = hre;

describe("AnalyticsRegistry", function () {
  let analyticsRegistry;
  let owner, analyst1, analyst2;

  beforeEach(async function () {
    [owner, analyst1, analyst2] = await ethers.getSigners();
    const AnalyticsRegistry = await ethers.getContractFactory("AnalyticsRegistry");
    analyticsRegistry = await AnalyticsRegistry.deploy();
    await analyticsRegistry.deployed();
  });

  describe("registerAnalytics", function () {
    it("Should register analytics and emit event", async function () {
      const sourceCID = "QmSource123";
      const resultCID = "QmResult123";
      const analysisType = "descriptive";

      const tx = await analyticsRegistry.connect(analyst1).registerAnalytics(sourceCID, resultCID, analysisType);
      const receipt = await tx.wait();

      expect(receipt.events[0].event).to.equal("AnalyticsRegistered");
      expect(receipt.events[0].args.analyst).to.equal(analyst1.address);
      expect(receipt.events[0].args.sourceCID).to.equal(sourceCID);
      expect(receipt.events[0].args.resultCID).to.equal(resultCID);
      expect(receipt.events[0].args.analysisType).to.equal(analysisType);
      expect(receipt.events[0].args.timestamp).to.be.gt(0);
    });

    it("Should reject if source CID is empty", async function () {
      await expect(
        analyticsRegistry.connect(analyst1).registerAnalytics("", "QmResult123", "descriptive")
      ).to.be.revertedWith("Invalid source CID");
    });

    it("Should reject if result CID is empty", async function () {
      await expect(
        analyticsRegistry.connect(analyst1).registerAnalytics("QmSource123", "", "descriptive")
      ).to.be.revertedWith("Invalid result CID");
    });

    it("Should reject if analysis type is empty", async function () {
      await expect(
        analyticsRegistry.connect(analyst1).registerAnalytics("QmSource123", "QmResult123", "")
      ).to.be.revertedWith("Invalid analysis type");
    });

    it("Should reject duplicate result CID", async function () {
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource1", "QmDupeResult", "descriptive");

      await expect(
        analyticsRegistry.connect(analyst1).registerAnalytics("QmSource2", "QmDupeResult", "graphical")
      ).to.be.revertedWith("Result CID already registered");
    });

    it("Should allow same dataset with different result CIDs", async function () {
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSameSource", "QmResultX", "descriptive");
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSameSource", "QmResultY", "graphical");

      const results = await analyticsRegistry.getAnalyticsForDataset("QmSameSource");
      expect(results.length).to.equal(2);
    });
  });

  describe("getAnalyticsForDataset", function () {
    it("Should return empty array for unknown dataset", async function () {
      const results = await analyticsRegistry.getAnalyticsForDataset("QmUnknown");
      expect(results.length).to.equal(0);
    });

    it("Should return all result CIDs for a source dataset", async function () {
      const sourceCID = "QmSource456";
      
      await analyticsRegistry.connect(analyst1).registerAnalytics(sourceCID, "QmResultA", "descriptive");
      await analyticsRegistry.connect(analyst2).registerAnalytics(sourceCID, "QmResultB", "graphical");

      const results = await analyticsRegistry.getAnalyticsForDataset(sourceCID);
      
      expect(results.length).to.equal(2);
      expect(results[0]).to.equal("QmResultA");
      expect(results[1]).to.equal("QmResultB");
    });
  });

  describe("getMyAnalytics and Pagination", function () {
    beforeEach(async function () {
      // Register 5 records for analyst1
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource1", "QmResult1", "descriptive");
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource2", "QmResult2", "graphical");
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource3", "QmResult3", "inferential");
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource4", "QmResult4", "descriptive");
      await analyticsRegistry.connect(analyst1).registerAnalytics("QmSource5", "QmResult5", "graphical");
      
      // Register 1 record for analyst2
      await analyticsRegistry.connect(analyst2).registerAnalytics("QmSourceA", "QmResultA", "descriptive");
    });

    it("Should return correct count", async function () {
      expect(await analyticsRegistry.connect(analyst1).getMyAnalyticsCount()).to.equal(5);
      expect(await analyticsRegistry.connect(analyst2).getMyAnalyticsCount()).to.equal(1);
    });

    it("Should return correct paginated results (offset 0, limit 2)", async function () {
      const results = await analyticsRegistry.connect(analyst1).getMyAnalytics(0, 2);
      expect(results.length).to.equal(2);
      expect(results[0].sourceCID).to.equal("QmSource1");
      expect(results[1].sourceCID).to.equal("QmSource2");
    });

    it("Should return correct paginated results (offset 2, limit 2)", async function () {
      const results = await analyticsRegistry.connect(analyst1).getMyAnalytics(2, 2);
      expect(results.length).to.equal(2);
      expect(results[0].sourceCID).to.equal("QmSource3");
      expect(results[1].sourceCID).to.equal("QmSource4");
    });

    it("Should handle limit exceeding remaining records", async function () {
      const results = await analyticsRegistry.connect(analyst1).getMyAnalytics(4, 10);
      expect(results.length).to.equal(1);
      expect(results[0].sourceCID).to.equal("QmSource5");
    });

    it("Should return empty array when offset is out of bounds", async function () {
      const results = await analyticsRegistry.connect(analyst1).getMyAnalytics(10, 2);
      expect(results.length).to.equal(0);
    });
    
    it("Should maintain isolation between wallets", async function () {
      const results = await analyticsRegistry.connect(analyst2).getMyAnalytics(0, 10);
      expect(results.length).to.equal(1);
      expect(results[0].sourceCID).to.equal("QmSourceA");
    });
  });
});
