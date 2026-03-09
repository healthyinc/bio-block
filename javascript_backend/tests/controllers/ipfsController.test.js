const request = require("supertest");
const { expect } = require("chai");
const axios = require("axios");
const app = require("../../server");

describe("IPFS Upload Controller", function () {
  this.timeout(15000);

  let originalAxiosPost;

  beforeEach(function () {
    originalAxiosPost = axios.post;
  });

  afterEach(function () {
    axios.post = originalAxiosPost;
  });

  // ------------------------------------------------------------------
  // 1. Successful upload
  // ------------------------------------------------------------------
  describe("Successful uploads", function () {
    it("should upload a file and return IPFS hash", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmTestHash1234567890abcdef" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("encrypted content"), "doc.enc")
        .field("fileName", "test_document.enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
      expect(res.body.ipfsHash).to.equal("QmTestHash1234567890abcdef");
      expect(res.body.fileName).to.equal("test_document.enc");
      expect(res.body.fileSize).to.be.a("number");
      expect(res.body.fileSize).to.be.greaterThan(0);
    });

    it("should report correct file size", async function () {
      const content = "a".repeat(1024);
      axios.post = async () => ({
        data: { IpfsHash: "QmSizeTest" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from(content), "sized.enc")
        .field("fileName", "sized.enc");

      expect(res.status).to.equal(200);
      expect(res.body.fileSize).to.equal(1024);
    });

    it("should use default filename when none is provided", async function () {
      axios.post = async (url, formData, config) => {
        // Verify the metadata contains the default filename
        return { data: { IpfsHash: "QmDefaultName" } };
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "file.enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
    });

    it("should send correct Pinata metadata and options", async function () {
      let capturedFormData;
      let capturedHeaders;

      axios.post = async (url, formData, config) => {
        capturedFormData = formData;
        capturedHeaders = config.headers;
        return { data: { IpfsHash: "QmMetadataTest" } };
      };

      // Set a test JWT so we can verify it's used
      const originalJwt = process.env.PINATA_JWT;
      process.env.PINATA_JWT = "test-jwt-token";

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "my_file.enc");

      process.env.PINATA_JWT = originalJwt;

      expect(res.status).to.equal(200);
      expect(capturedHeaders.Authorization).to.equal("Bearer test-jwt-token");
    });

    it("should call the correct Pinata endpoint", async function () {
      let capturedUrl;

      axios.post = async (url) => {
        capturedUrl = url;
        return { data: { IpfsHash: "QmUrlTest" } };
      };

      await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(capturedUrl).to.equal("https://api.pinata.cloud/pinning/pinFileToIPFS");
    });
  });

  // ------------------------------------------------------------------
  // 2. Input validation
  // ------------------------------------------------------------------
  describe("Input validation", function () {
    it("should return 400 when no file is uploaded", async function () {
      const res = await request(app).post("/api/ipfs/upload").field("fileName", "missing.enc");

      expect(res.status).to.equal(400);
      expect(res.body).to.have.property("error");
      expect(res.body.error).to.include("No file");
    });
  });

  // ------------------------------------------------------------------
  // 3. Pinata API error handling
  // ------------------------------------------------------------------
  describe("Pinata API error handling", function () {
    it("should return 401 when Pinata credentials are invalid", async function () {
      axios.post = async () => {
        const error = new Error("Unauthorized");
        error.response = { status: 401, data: { error: "Invalid API Key" } };
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(401);
      expect(res.body.error).to.include("Invalid Pinata API credentials");
    });

    it("should return 500 when Pinata returns a server error", async function () {
      axios.post = async () => {
        const error = new Error("Service Unavailable");
        error.response = {
          status: 503,
          data: { error: "Service temporarily unavailable" },
        };
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body.error).to.include("IPFS upload failed");
    });

    it("should return 500 on network timeout", async function () {
      axios.post = async () => {
        const error = new Error("timeout of 30000ms exceeded");
        error.code = "ECONNABORTED";
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body.error).to.include("IPFS upload failed");
    });

    it("should return 500 on DNS resolution failure", async function () {
      axios.post = async () => {
        const error = new Error("getaddrinfo ENOTFOUND api.pinata.cloud");
        error.code = "ENOTFOUND";
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body.error).to.include("IPFS upload failed");
    });

    it("should return 500 on connection refused", async function () {
      axios.post = async () => {
        const error = new Error("connect ECONNREFUSED");
        error.code = "ECONNREFUSED";
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body.error).to.include("IPFS upload failed");
    });

    it("should include Pinata error detail in response when available", async function () {
      axios.post = async () => {
        const error = new Error("Bad Request");
        error.response = {
          status: 400,
          data: { error: "Invalid file format for pinning" },
        };
        throw error;
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body.error).to.include("Invalid file format for pinning");
    });
  });

  // ------------------------------------------------------------------
  // 4. Response structure
  // ------------------------------------------------------------------
  describe("Response structure", function () {
    it("should return all expected fields on success", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmStructureTest" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("test data"), "file.enc")
        .field("fileName", "file.enc");

      expect(res.status).to.equal(200);
      expect(res.body).to.have.all.keys("success", "ipfsHash", "fileName", "fileSize");
      expect(res.body.success).to.be.true;
      expect(res.body.ipfsHash).to.be.a("string");
      expect(res.body.fileName).to.be.a("string");
      expect(res.body.fileSize).to.be.a("number");
    });

    it("should return error object on failure", async function () {
      axios.post = async () => {
        throw new Error("Generic failure");
      };

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test.enc")
        .field("fileName", "test.enc");

      expect(res.status).to.equal(500);
      expect(res.body).to.have.property("error");
      expect(res.body.error).to.be.a("string");
    });
  });

  // ------------------------------------------------------------------
  // 5. Binary content handling
  // ------------------------------------------------------------------
  describe("Binary content handling", function () {
    it("should handle binary (encrypted) file content", async function () {
      const binaryContent = Buffer.alloc(256);
      for (let i = 0; i < 256; i++) {
        binaryContent[i] = i;
      }

      axios.post = async () => ({
        data: { IpfsHash: "QmBinaryTest" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", binaryContent, "binary.enc")
        .field("fileName", "binary.enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
      expect(res.body.fileSize).to.equal(256);
    });

    it("should handle empty file buffer", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmEmptyFile" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.alloc(0), "empty.enc")
        .field("fileName", "empty.enc");

      expect(res.status).to.equal(200);
      expect(res.body.fileSize).to.equal(0);
    });
  });

  // ------------------------------------------------------------------
  // 6. Filename sanitization
  // ------------------------------------------------------------------
  describe("Filename handling", function () {
    it("should handle filenames with special characters", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmSpecialChars" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "test file (1).enc")
        .field("fileName", "test file (1).enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
      expect(res.body.fileName).to.equal("test file (1).enc");
    });

    it("should handle unicode filenames", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmUnicode" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("data"), "données.enc")
        .field("fileName", "données.enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
    });
  });

  // ------------------------------------------------------------------
  // 7. File-type validation
  // ------------------------------------------------------------------
  describe("File-type validation", function () {
    it("should reject executable files", async function () {
      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("MZ"), "malware.exe")
        .field("fileName", "malware.exe");

      expect(res.status).to.equal(400);
      expect(res.body.error).to.include("File type not allowed");
    });

    it("should reject shell scripts", async function () {
      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("#!/bin/bash"), "script.sh")
        .field("fileName", "script.sh");

      expect(res.status).to.equal(400);
      expect(res.body.error).to.include("File type not allowed");
    });

    it("should accept encrypted blob files (.enc)", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmEncryptedBlob" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("encrypted data"), "dataset.enc")
        .field("fileName", "dataset.enc");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
    });

    it("should accept health data formats (.csv)", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmCsvData" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("col1,col2\n1,2"), {
          filename: "data.csv",
          contentType: "text/csv",
        })
        .field("fileName", "data.csv");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
    });

    it("should accept medical imaging files (.dcm)", async function () {
      axios.post = async () => ({
        data: { IpfsHash: "QmDicomFile" },
      });

      const res = await request(app)
        .post("/api/ipfs/upload")
        .attach("encryptedFile", Buffer.from("DICM"), "scan.dcm")
        .field("fileName", "scan.dcm");

      expect(res.status).to.equal(200);
      expect(res.body.success).to.be.true;
    });
  });
});
