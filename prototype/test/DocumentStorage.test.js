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
      ).to.be.revertedWith("Exact price required");
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

  describe("hasAccess", function () {
    it("Should return true for document owner", async function () {
      const ipfsHash = "QmAccess100";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      expect(await documentStorage.hasAccess(ipfsHash, owner.address)).to.equal(true);
    });

    it("Should return false for non-buyer before purchase", async function () {
      const ipfsHash = "QmAccess200";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      expect(await documentStorage.hasAccess(ipfsHash, buyer.address)).to.equal(false);
    });

    it("Should return true for buyer after purchase", async function () {
      const ipfsHash = "QmAccess300";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");
      await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });

      expect(await documentStorage.hasAccess(ipfsHash, buyer.address)).to.equal(true);
    });

    it("Should set hasPurchased flag after purchase", async function () {
      const ipfsHash = "QmAccess400";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      expect(await documentStorage.hasPurchased(ipfsHash, buyer.address)).to.equal(false);
      await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });
      expect(await documentStorage.hasPurchased(ipfsHash, buyer.address)).to.equal(true);
    });

    it("Should return false for unrelated address", async function () {
      const ipfsHash = "QmAccess500";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");
      await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });

      expect(await documentStorage.hasAccess(ipfsHash, addr1.address)).to.equal(false);
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

  describe("storeDocument security", function () {
    it("Should reject storing a document with an already registered hash", async function () {
      const ipfsHash = "QmDuplicate123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "original");

      await expect(
        documentStorage
          .connect(buyer)
          .storeDocument(ipfsHash, ethers.utils.parseEther("0.1"), "hijacked")
      ).to.be.revertedWith("Document already exists");

      const storedOwner = await documentStorage.documentOwners(ipfsHash);
      expect(storedOwner).to.equal(owner.address);
    });

    it("Should allow re-registering a hash after the original is deleted", async function () {
      const ipfsHash = "QmReRegister123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "original");
      await documentStorage.deleteDocument(ipfsHash);

      await documentStorage
        .connect(buyer)
        .storeDocument(ipfsHash, ethers.utils.parseEther("2.0"), "new owner");

      const newOwner = await documentStorage.documentOwners(ipfsHash);
      expect(newOwner).to.equal(buyer.address);
    });
  });

  describe("purchaseDocument security", function () {
    it("Should reject purchase of unregistered document", async function () {
      await expect(
        documentStorage
          .connect(buyer)
          .purchaseDocument("QmNeverRegistered", { value: ethers.utils.parseEther("1.0") })
      ).to.be.revertedWith("Document does not exist");
    });

    it("Should reject owner self-purchase", async function () {
      const ipfsHash = "QmSelfBuy123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      await expect(documentStorage.purchaseDocument(ipfsHash, { value: price })).to.be.revertedWith(
        "Cannot purchase own document"
      );
    });

    it("Should reject purchase with incorrect price", async function () {
      const ipfsHash = "QmExactPrice123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      await expect(
        documentStorage
          .connect(buyer)
          .purchaseDocument(ipfsHash, { value: ethers.utils.parseEther("2.0") })
      ).to.be.revertedWith("Exact price required");
    });

    it("Should credit exactly the document price to seller earnings", async function () {
      const ipfsHash = "QmEarnings123";
      const price = ethers.utils.parseEther("1.5");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");
      await documentStorage.connect(buyer).purchaseDocument(ipfsHash, { value: price });

      const sellerEarnings = await documentStorage.earnings(owner.address);
      expect(sellerEarnings).to.equal(price);
    });
  });

  describe("deleteDocument cleanup", function () {
    it("Should remove document from user's document list", async function () {
      const ipfsHash = "QmCleanup123";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(ipfsHash, price, "metadata");

      let docs = await documentStorage.getMyDocuments();
      expect(docs).to.include(ipfsHash);

      await documentStorage.deleteDocument(ipfsHash);

      docs = await documentStorage.getMyDocuments();
      expect(docs).to.not.include(ipfsHash);
    });

    it("Should not affect other documents in list when deleting", async function () {
      const hash1 = "QmMulti1";
      const hash2 = "QmMulti2";
      const hash3 = "QmMulti3";
      const price = ethers.utils.parseEther("1.0");

      await documentStorage.storeDocument(hash1, price, "doc1");
      await documentStorage.storeDocument(hash2, price, "doc2");
      await documentStorage.storeDocument(hash3, price, "doc3");

      await documentStorage.deleteDocument(hash2);

      const docs = await documentStorage.getMyDocuments();
      expect(docs.length).to.equal(2);
      expect(docs).to.include(hash1);
      expect(docs).to.include(hash3);
      expect(docs).to.not.include(hash2);
    });
  });
});
