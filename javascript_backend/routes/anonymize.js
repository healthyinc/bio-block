const express = require("express");
const router = express.Router();
const { anonymizeFile, upload } = require("../controllers/anonymizeController");

// POST /api/anonymize - Anonymize Excel files
router.post("/", upload.single("file"), anonymizeFile);

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
