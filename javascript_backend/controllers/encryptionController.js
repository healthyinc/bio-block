const crypto = require('crypto');
const multer = require('multer');

const getEncryptionKey = (userId) => {
    const salt = process.env.ENCRYPTION_SALT || 'default-salt-change-in-production';
    const key = crypto.pbkdf2Sync(userId + salt, salt, 100000, 32, 'sha512');
    return key;
};

const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 * 1024
    }
});

const encryptFile = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file provided for encryption' 
            });
        }

        const userId = req.user.walletAddress;
        const key = getEncryptionKey(userId);
        const iv = crypto.randomBytes(16);

        const cipher = crypto.createCipher('aes-256-gcm', key);
        cipher.setAAD(Buffer.from(userId));
        
        const encrypted = Buffer.concat([
            cipher.update(req.file.buffer),
            cipher.final()
        ]);

        const authTag = cipher.getAuthTag();

        const result = Buffer.concat([
            iv,
            authTag,
            encrypted
        ]);

        res.json({
            success: true,
            encryptedData: result.toString('base64'),
            algorithm: 'aes-256-gcm',
            keyDerivation: 'pbkdf2'
        });

    } catch (error) {
        console.error('Encryption error:', error);
        res.status(500).json({
            error: 'Encryption failed: ' + error.message
        });
    }
};

const decryptFile = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No encrypted file provided' 
            });
        }

        const userId = req.user.walletAddress;
        const key = getEncryptionKey(userId);

        const encryptedBuffer = req.file.buffer;
        const iv = encryptedBuffer.slice(0, 16);
        const authTag = encryptedBuffer.slice(16, 32);
        const encrypted = encryptedBuffer.slice(32);

        const decipher = crypto.createDecipher('aes-256-gcm', key);
        decipher.setAAD(Buffer.from(userId));
        decipher.setAuthTag(authTag);

        const decrypted = Buffer.concat([
            decipher.update(encrypted),
            decipher.final()
        ]);

        res.json({
            success: true,
            decryptedData: decrypted.toString('base64')
        });

    } catch (error) {
        console.error('Decryption error:', error);
        res.status(500).json({
            error: 'Decryption failed: ' + error.message
        });
    }
};

const encryptFileStream = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file provided for streaming encryption' 
            });
        }

        const userId = req.user.walletAddress;
        const key = getEncryptionKey(userId);
        const chunkSize = parseInt(req.body.chunkSize) || 1024 * 1024;

        const fileBuffer = req.file.buffer;
        const totalChunks = Math.ceil(fileBuffer.length / chunkSize);
        const encryptedChunks = [];

        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, fileBuffer.length);
            const chunk = fileBuffer.slice(start, end);

            const iv = crypto.randomBytes(16);
            const cipher = crypto.createCipher('aes-256-gcm', key);
            cipher.setAAD(Buffer.from(`${userId}-chunk-${i}`));

            const encrypted = Buffer.concat([
                cipher.update(chunk),
                cipher.final()
            ]);

            const authTag = cipher.getAuthTag();

            encryptedChunks.push({
                iv: iv.toString('base64'),
                authTag: authTag.toString('base64'),
                data: encrypted.toString('base64'),
                chunkIndex: i
            });
        }

        const encryptedPackage = {
            metadata: {
                totalChunks: totalChunks,
                originalSize: fileBuffer.length,
                chunkSize: chunkSize,
                algorithm: 'aes-256-gcm',
                keyDerivation: 'pbkdf2'
            },
            chunks: encryptedChunks
        };

        const result = Buffer.from(JSON.stringify(encryptedPackage));

        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', result.length);
        res.send(result);

    } catch (error) {
        console.error('Streaming encryption error:', error);
        res.status(500).json({
            error: 'Streaming encryption failed: ' + error.message
        });
    }
};

const decryptFileStream = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No encrypted file provided for streaming decryption' 
            });
        }

        const userId = req.user.walletAddress;
        const key = getEncryptionKey(userId);

        const encryptedPackage = JSON.parse(req.file.buffer.toString());
        const { metadata, chunks } = encryptedPackage;

        const decryptedChunks = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            
            const iv = Buffer.from(chunk.iv, 'base64');
            const authTag = Buffer.from(chunk.authTag, 'base64');
            const encrypted = Buffer.from(chunk.data, 'base64');

            const decipher = crypto.createDecipher('aes-256-gcm', key);
            decipher.setAAD(Buffer.from(`${userId}-chunk-${chunk.chunkIndex}`));
            decipher.setAuthTag(authTag);

            const decrypted = Buffer.concat([
                decipher.update(encrypted),
                decipher.final()
            ]);

            decryptedChunks.push(decrypted);
        }

        const result = Buffer.concat(decryptedChunks);

        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', result.length);
        res.send(result);

    } catch (error) {
        console.error('Streaming decryption error:', error);
        res.status(500).json({
            error: 'Streaming decryption failed: ' + error.message
        });
    }
};

module.exports = {
    encryptFile,
    decryptFile,
    encryptFileStream,
    decryptFileStream,
    upload
};