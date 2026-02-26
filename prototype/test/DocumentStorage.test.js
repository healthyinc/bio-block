const hre = require("hardhat");
const { expect } = require("chai");
const { ethers } = hre;

describe("DocumentStorage", function () {
  let documentStorage;
  let owner, buyer, addr1;

  beforeEach(async function () {
    [owner, buyer, addr1] = await ethers.getSigners();
    const DocumentStorage = await ethers.getContractFactory("DocumentStorage");
    documentStorage = await DocumentStorage.deploy();
    await documentStorage.deployed();
  });

  describe("storeDocument", function () {
    it("Should store a document with metadata", async function () {
      const ipfsHash = "QmTest123";
      const price = ethers.utils.parseEther("1.0");
      const metadata = "tags: medical, format: DICOM";

      const tx = await documentStorage.storeDocument(ipfsHash, price, metadata);
      const receipt = await tx.wait();

      expect(receipt.events[0].event).to.equal("DocumentStored");
      expect(receipt.events[0].args.owner).to.equal(owner.address);
      expect(receipt.events[0].args.ipfsHash).to.equal(ipfsHash);
    });

    it("Should retrieve stored document metadata", async function () {
      const ipfsHash = "QmTest456";
      const metadata = "description: test doc";

      await documentStorage.storeDocument(ipfsHash, ethers.utils.parseEther("0.5"), metadata);
      const retrieved = await documentStorage.getMetadata(ipfsHash);

      expect(retrieved).to.equal(metadata);
    });
  });

  describe("updateMetadata", function () {
    it("Should update document metadata", async function () {
      const ipfsHash = "QmTest789";
      await documentStorage.storeDocument(ipfsHash, ethers.utils.parseEther("1.0"), "old metadata");

      const newPrice = ethers.utils.parseEther("2.0");
      const newMetadata = "updated metadata";

      const tx = await documentStorage.updateMetadata(ipfsHash, newPrice, newMetadata);
      const receipt = await tx.wait();

      expect(receipt.events[0].event).to.equal("MetadataUpdated");

      const price = await documentStorage.documentPrices(ipfsHash);
      const metadata = await documentStorage.getMetadata(ipfsHash);

      expect(price.toString()).to.equal(newPrice.toString());
      expect(metadata).to.equal(newMetadata);
    });

    it("Should only allow owner to update", async function () {
      const ipfsHash = "QmTest999";
      await documentStorage.storeDocument(ipfsHash, ethers.utils.parseEther("1.0"), "metadata");

      await expect(
        documentStorage
          .connect(buyer)
          .updateMetadata(ipfsHash, ethers.utils.parseEther("1.0"), "hacked")
      ).to.be.revertedWith("Only owner can update");
    });

    it("Should reject update for non-existent document", async function () {
      await expect(
        documentStorage.updateMetadata("QmFake", ethers.utils.parseEther("1.0"), "metadata")
      ).to.be.revertedWith("Document does not exist");
    });
  });

  describe("deleteDocument", function () {
    it("Should delete a document", async function () {
      const ipfsHash = "QmDelete123";
      await documentStorage.storeDocument(ipfsHash, ethers.utils.parseEther("1.0"), "metadata");

      const tx = await documentStorage.deleteDocument(ipfsHash);
      const receipt = await tx.wait();

      expect(receipt.events[0].event).to.equal("DocumentDeleted");

      const owner_after = await documentStorage.documentOwners(ipfsHash);
      const price_after = await documentStorage.documentPrices(ipfsHash);
      const metadata_after = await documentStorage.getMetadata(ipfsHash);

      expect(owner_after).to.equal(ethers.constants.AddressZero);
      expect(price_after).to.equal(ethers.constants.Zero);
      expect(metadata_after).to.equal("");
    });

    it("Should only allow owner to delete", async function () {
      const ipfsHash = "QmDelete456";
      await documentStorage.storeDocument(ipfsHash, ethers.utils.parseEther("1.0"), "metadata");

      await expect(documentStorage.connect(buyer).deleteDocument(ipfsHash)).to.be.revertedWith(
        "Only owner can delete"
      );
    });

    it("Should reject delete for non-existent document", async function () {
      await expect(documentStorage.deleteDocument("QmFakeDelete")).to.be.revertedWith(
        "Document does not exist"
      );
    });
  });

  describe("purchaseDocument", function () {
    it("Should purchase a document and emit event", async function () {
      const ipfsHash = "QmBuy123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      const tx = await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });
      const receipt = await tx.wait();

      expect(receipt.events[0].event).to.equal("DocumentPurchased");
      expect(receipt.events[0].args.buyer).to.equal(buyer.address);
    });

    it("Should reject purchase with insufficient payment", async function () {
      const ipfsHash = "QmBuy456";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      await expect(
        documentStorage
          .connect(buyer)
          .purchaseDocument(ipfsHash, { value: ethers.utils.parseEther("0.5") })
      ).to.be.revertedWith("Insufficient payment");
    });

    it("Should reject purchase of deleted document", async function () {
      const ipfsHash = "QmBuy789";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");
      await documentStorage.deleteDocument(ipfsHash);

      await expect(
        documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price })
      ).to.be.revertedWith("Document does not exist");
    });
  });

  describe("withdrawEarnings", function () {
    it("Should withdraw earnings", async function () {
      const ipfsHash = "QmWithdraw123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");
      await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });

      const balanceBefore = await ethers.provider.getBalance(owner.address);
      const tx = await documentStorage.withdrawEarnings();
      const receipt = await tx.wait();
      const gasUsed = receipt.gasUsed.mul(receipt.effectiveGasPrice);

      const earnings_after = await documentStorage.earnings(owner.address);
      expect(earnings_after).to.equal(ethers.constants.Zero);
    });
  });
});
