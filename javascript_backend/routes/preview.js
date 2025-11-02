const express = require('express');
const router = express.Router();
const multer = require('multer');
const previewController = require('../controllers/previewController');

// Setup multer for in-memory file storage. This is how we get req.file
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Define the route:
// When a POST request comes to /image,
// 1. Use 'upload.single('file')' to handle the file
// 2. Pass it to 'previewController.getImagePreview'
router.post('/image', upload.single('file'), previewController.getImagePreview);

module.exports = router;