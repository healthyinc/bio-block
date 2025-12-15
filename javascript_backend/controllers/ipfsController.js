const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

// Configure multer for memory storage
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: { 
        fileSize: 10 * 1024 * 1024 * 1024 // 10GB limit
    }
});

const uploadToIPFS = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file uploaded.' 
            });
        }

        const { fileName } = req.body;
        const encryptedBuffer = req.file.buffer;

        // Create form data for Pinata
        const formData = new FormData();
        formData.append('file', encryptedBuffer, {
            filename: fileName || 'encrypted_file',
            contentType: 'application/octet-stream'
        });

        // Pinata metadata
        const pinataMetadata = JSON.stringify({
            name: fileName || 'Encrypted Document',
            keyvalues: {
                encrypted: 'true',
                uploadedAt: new Date().toISOString()
            }
        });
        formData.append('pinataMetadata', pinataMetadata);

        // Pinata options
        const pinataOptions = JSON.stringify({
            cidVersion: 0,
        });
        formData.append('pinataOptions', pinataOptions);

        // Upload to Pinata
        const pinataResponse = await axios.post(
            'https://api.pinata.cloud/pinning/pinFileToIPFS',
            formData,
            {
                maxBodyLength: 'Infinity',
                headers: {
                    'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
                    'Authorization': `Bearer ${process.env.PINATA_JWT}`,
                    ...formData.getHeaders()
                }
            }
        );

        const ipfsHash = pinataResponse.data.IpfsHash;

        res.json({
            success: true,
            ipfsHash: ipfsHash,
            fileName: fileName,
            fileSize: encryptedBuffer.length
        });

    } catch (error) {
        if (error.response?.status === 401) {
            return res.status(401).json({ 
                error: 'Invalid Pinata API credentials' 
            });
        }
        
        res.status(500).json({ 
            error: 'IPFS upload failed: ' + (error.response?.data?.error || error.message)
        });
    }
};

module.exports = {
    uploadToIPFS,
    upload
};
