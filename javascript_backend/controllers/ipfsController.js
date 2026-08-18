const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

const KEYS_FILE = path.join(__dirname, '../data/keys.json');

// Ensure data dir exists
if (!fs.existsSync(path.dirname(KEYS_FILE))) {
    fs.mkdirSync(path.dirname(KEYS_FILE), { recursive: true });
}
if (!fs.existsSync(KEYS_FILE)) {
    fs.writeFileSync(KEYS_FILE, JSON.stringify({}));
}

// Contract configuration
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS || '0x5FbDB2315678afecb367f032d93F642f64180aa3';
const CONTRACT_ABI = [
    "function checkAccess(string memory ipfsHash, address user) public view returns (bool)"
];

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

        const { fileName, documentKey } = req.body;
        const encryptedBuffer = req.file.buffer;

        if (!documentKey) {
            return res.status(400).json({ error: 'documentKey is required for encryption.' });
        }


        console.log('Uploading to IPFS:', {
            fileName,
            fileSize: encryptedBuffer.length
        });

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

        let ipfsHash;
        
        if (!process.env.PINATA_JWT || process.env.PINATA_JWT === 'dummy_jwt_token') {
            console.log("Using mock IPFS upload (no real Pinata credentials)");
            // Generate a fake IPFS hash for local testing
            const crypto = require('crypto');
            ipfsHash = 'QmMock' + crypto.randomBytes(20).toString('hex');
            // Mock a small delay
            await new Promise(resolve => setTimeout(resolve, 1000));
        } else {
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

            ipfsHash = pinataResponse.data.IpfsHash;
        }

        // Save the key
        const keys = JSON.parse(fs.readFileSync(KEYS_FILE, 'utf8'));
        keys[ipfsHash] = documentKey;
        fs.writeFileSync(KEYS_FILE, JSON.stringify(keys, null, 2));

        console.log('IPFS upload and key storage successful:', { ipfsHash, fileName });

        res.json({
            success: true,
            ipfsHash: ipfsHash,
            fileName: fileName,
            fileSize: encryptedBuffer.length
        });

    } catch (error) {
        console.error('IPFS upload error:', error.response?.data || error.message);

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

const getDocumentKey = async (req, res) => {
    try {
        const { ipfsHash } = req.params;
        const { buyerAddress } = req.query;

        if (!buyerAddress) {
            return res.status(400).json({ error: 'buyerAddress is required' });
        }

        // Verify access on the blockchain
        const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545'); // Default Hardhat local network
        const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);

        const hasAccess = await contract.checkAccess(ipfsHash, buyerAddress);

        if (!hasAccess) {
            return res.status(403).json({ error: 'Access denied: You have not purchased this document or you are not the owner' });
        }

        // Retrieve key
        const keys = JSON.parse(fs.readFileSync(KEYS_FILE, 'utf8'));
        const documentKey = keys[ipfsHash];

        if (!documentKey) {
            return res.status(404).json({ error: 'Key not found for this document' });
        }

        res.json({ documentKey });

    } catch (error) {
        console.error('Key retrieval error:', error);
        res.status(500).json({ error: 'Failed to verify access or retrieve key. Make sure the blockchain network is running.' });
    }
};

module.exports = {
    uploadToIPFS,
    getDocumentKey,
    upload
};
