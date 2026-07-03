const express = require("express");
const router = express.Router();
const { uploadToIPFS, upload } = require("../controllers/ipfsController");
const { uploadAnalyticsResult } = require("../controllers/analyticsIpfsController");

router.post("/upload", upload.single("encryptedFile"), uploadToIPFS);

router.post("/upload-analytics-result", express.json(), uploadAnalyticsResult);

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
