const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/auth');
const { 
    encryptFile, 
    decryptFile, 
    encryptFileStream, 
    decryptFileStream, 
    upload 
} = require('../controllers/encryptionController');

router.use(authenticateToken);

router.post('/encrypt', upload.single('file'), encryptFile);
router.post('/decrypt', upload.single('encryptedFile'), decryptFile);
router.post('/encrypt-stream', upload.single('file'), encryptFileStream);
router.post('/decrypt-stream', upload.single('encryptedFile'), decryptFileStream);

module.exports = router;