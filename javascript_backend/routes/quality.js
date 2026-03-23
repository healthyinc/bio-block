const express = require("express");
const router = express.Router();
const qualityController = require("../controllers/qualityController");

// Route for dataset quality profiling
router.post(
  "/profile",
  qualityController.upload.single("file"),
  qualityController.analyzeDatasetQuality
);

module.exports = router;
