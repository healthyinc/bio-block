const express = require('express');
const router = express.Router();
const multer = require('multer');
const previewController = require('../controllers/previewController');

// Setup multer for in-memory file storage. This is how we get req.file
const storage = multer.memoryStorage();
const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB limit for preview files
  }
});

// Define the routes:
// When a POST request comes to /image, /spreadsheet, /pdf, or /dicom,
// 1. Use 'upload.single('file')' to handle the file
// 2. Pass it to the corresponding preview controller
router.post('/image', upload.single('file'), previewController.getImagePreview);
router.post('/spreadsheet', upload.single('file'), previewController.getSpreadsheetPreview);
router.post('/pdf', upload.single('file'), previewController.getPdfPreview);
router.post('/dicom', upload.single('file'), previewController.getDicomPreview);

module.exports = router;