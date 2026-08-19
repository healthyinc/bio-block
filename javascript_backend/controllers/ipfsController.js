const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const os = require("os");
const crypto = require("crypto");
const path = require("path");
const { ethers } = require("ethers");

const KEYS_FILE = path.join(__dirname, "../data/keys.json");

// Ensure data dir exists
if (!fs.existsSync(path.dirname(KEYS_FILE))) {
  fs.mkdirSync(path.dirname(KEYS_FILE), { recursive: true });
}
if (!fs.existsSync(KEYS_FILE)) {
  fs.writeFileSync(KEYS_FILE, JSON.stringify({}));
}

const PREVIEW_KEYS_FILE = path.join(__dirname, "../data/preview_keys.json");
if (!fs.existsSync(PREVIEW_KEYS_FILE)) {
  fs.writeFileSync(PREVIEW_KEYS_FILE, JSON.stringify({}));
}

// Contract configuration
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS || "0x5FbDB2315678afecb367f032d93F642f64180aa3";
const CONTRACT_ABI = [
  "function checkAccess(string memory ipfsHash, address user) public view returns (bool)"
];

// Configure multer for disk storage with file-type validation
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, os.tmpdir());
  },
  filename: (req, file, cb) => {
    const uniqueName = `upload-${crypto.randomUUID()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "application/octet-stream",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "text/csv",
      "application/csv",
      "application/vnd.oasis.opendocument.spreadsheet",
      "text/tab-separated-values",
      "image/jpeg",
      "image/png",
      "application/pdf",
      "application/dicom",
    ];
    const allowedExtensions = /\.(enc|xlsx|xls|csv|ods|tsv|jpg|jpeg|png|pdf|dcm|nii|nii\.gz)$/i;
    if (allowedMimeTypes.includes(file.mimetype) || allowedExtensions.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error("File type not allowed."), false);
    }
  },
  limits: { fileSize: 2 * 1024 * 1024 * 1024 },
});

const uploadToIPFS = async (req, res) => {
  const cleanupTempFile = () => {
    if (req.file && req.file.path) {
      fs.unlink(req.file.path, (err) => {
        if (err && err.code !== "ENOENT") console.error("Failed to clean up temp file:", err);
      });
    }
  };

  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    if (!process.env.PINATA_JWT || process.env.PINATA_JWT === "your_pinata_jwt_here") {
      console.log("Mocking Pinata upload for local testing...");
      const mockHash = "QmMock" + crypto.randomBytes(20).toString("hex");
      const documentKey = req.body.documentKey || crypto.randomBytes(32).toString("hex");

      if (req.body.isPreview === "true") {
        const previewKeys = JSON.parse(fs.readFileSync(PREVIEW_KEYS_FILE, "utf8"));
        previewKeys[mockHash] = documentKey;
        fs.writeFileSync(PREVIEW_KEYS_FILE, JSON.stringify(previewKeys, null, 2));
      } else {
        const keys = JSON.parse(fs.readFileSync(KEYS_FILE, "utf8"));
        keys[mockHash] = documentKey;
        fs.writeFileSync(KEYS_FILE, JSON.stringify(keys, null, 2));
      }

      // Save the mock file locally so it can be downloaded later
      const mockDir = path.join(__dirname, "../data/mock_ipfs");
      if (!fs.existsSync(mockDir)) fs.mkdirSync(mockDir, { recursive: true });
      fs.copyFileSync(req.file.path, path.join(mockDir, mockHash));

      cleanupTempFile();
      return res.status(200).json({
        success: true,
        ipfsHash: mockHash,
        documentKey: documentKey,
        mocked: true,
      });
    }

    const fileName = req.body.fileName || req.file.originalname;
    const documentKey = req.body.documentKey;

    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append("file", fileStream, {
      filename: fileName || "encrypted_file",
      contentType: "application/octet-stream",
    });

    const pinataMetadata = JSON.stringify({
      name: fileName || "Encrypted Document",
      keyvalues: { encrypted: "true", uploadedAt: new Date().toISOString() },
    });
    formData.append("pinataMetadata", pinataMetadata);

    const pinataOptions = JSON.stringify({ cidVersion: 0 });
    formData.append("pinataOptions", pinataOptions);

    const pinataResponse = await axios.post(
      "https://api.pinata.cloud/pinning/pinFileToIPFS",
      formData,
      {
        maxBodyLength: "Infinity",
        headers: {
          "Content-Type": `multipart/form-data; boundary=${formData._boundary}`,
          Authorization: `Bearer ${process.env.PINATA_JWT}`,
          ...formData.getHeaders(),
        },
      }
    );

    const ipfsHash = pinataResponse.data.IpfsHash;

    if (documentKey) {
      if (req.body.isPreview === "true") {
        const previewKeys = JSON.parse(fs.readFileSync(PREVIEW_KEYS_FILE, "utf8"));
        previewKeys[ipfsHash] = documentKey;
        fs.writeFileSync(PREVIEW_KEYS_FILE, JSON.stringify(previewKeys, null, 2));
      } else {
        const keys = JSON.parse(fs.readFileSync(KEYS_FILE, "utf8"));
        keys[ipfsHash] = documentKey;
        fs.writeFileSync(KEYS_FILE, JSON.stringify(keys, null, 2));
      }
    }

    cleanupTempFile();
    res.json({
      success: true,
      ipfsHash: ipfsHash,
      fileName: fileName,
      fileSize: req.file.size,
    });
  } catch (error) {
    cleanupTempFile();
    console.error("IPFS upload error:", error.response?.data || error.message);
    if (error.response?.status === 401) {
      return res.status(401).json({ error: "Invalid Pinata API credentials" });
    }
    res.status(500).json({
      error: "IPFS upload failed: " + (error.response?.data?.error || error.message),
    });
  }
};

const getDocumentKey = async (req, res) => {
  try {
    const { ipfsHash } = req.params;
    const { buyerAddress } = req.query;

    if (!buyerAddress) return res.status(400).json({ error: "buyerAddress is required" });

    // Previews are public, no access check needed
    const previewKeys = JSON.parse(fs.readFileSync(PREVIEW_KEYS_FILE, "utf8"));
    if (previewKeys[ipfsHash]) {
      return res.json({ documentKey: previewKeys[ipfsHash] });
    }

    const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");
    const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);

    const hasAccess = await contract.checkAccess(ipfsHash, buyerAddress);
    if (!hasAccess) {
      return res.status(403).json({ error: "Access denied" });
    }

    const keys = JSON.parse(fs.readFileSync(KEYS_FILE, "utf8"));
    const documentKey = keys[ipfsHash];

    if (!documentKey) return res.status(404).json({ error: "Key not found" });

    res.json({ documentKey });
  } catch (error) {
    console.error("Key retrieval error:", error);
    res.status(500).json({ error: "Failed to verify access or retrieve key." });
  }
};

module.exports = {
  uploadToIPFS,
  getDocumentKey,
  upload,
};
