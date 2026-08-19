const express = require("express");
const router = express.Router();
const { uploadToIPFS, getDocumentKey, upload } = require("../controllers/ipfsController");
const { uploadAnalyticsResult } = require("../controllers/analyticsIpfsController");

router.post("/upload", upload.single("encryptedFile"), uploadToIPFS);

router.post("/upload-analytics-result", express.json(), uploadAnalyticsResult);

// GET /api/ipfs/key/:ipfsHash - Retrieve decryption key (requires payment verification)
router.get('/key/:ipfsHash', getDocumentKey);

const path = require("path");
const fs = require("fs");
router.get('/mock/:ipfsHash', (req, res) => {
  const filePath = path.join(__dirname, "../data/mock_ipfs", req.params.ipfsHash);
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).json({ error: "Mock file not found" });
  }
});

router.use((error, req, res, next) => {
  if (error.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      error: "File too large. Maximum size is 2GB.",
    });
  }
  if (error.message && error.message.startsWith("File type not allowed")) {
    return res.status(400).json({
      error: error.message,
    });
  }
  next(error);
});

module.exports = router;
