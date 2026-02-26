const express = require("express");
const router = express.Router();
const { uploadToIPFS, upload } = require("../controllers/ipfsController");

// POST /api/ipfs/upload - Upload encrypted files to IPFS
router.post("/upload", upload.single("encryptedFile"), uploadToIPFS);

// Error handling for multer
router.use((error, req, res, next) => {
  if (error.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      error: "File too large. Maximum size is 10GB.",
    });
  }
  next(error);
});

module.exports = router;
