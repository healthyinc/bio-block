const jwt = require('jsonwebtoken');
const { ethers } = require('ethers');

const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production';

const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ 
            error: 'Access denied. No token provided.' 
        });
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(403).json({ 
            error: 'Invalid or expired token.' 
        });
    }
};

const authenticateWallet = async (req, res) => {
    try {
        const { walletAddress, signature, message } = req.body;

        if (!walletAddress || !signature || !message) {
            return res.status(400).json({
                error: 'Missing required fields: walletAddress, signature, message'
            });
        }

        const recoveredAddress = ethers.utils.verifyMessage(message, signature);
        
        if (recoveredAddress.toLowerCase() !== walletAddress.toLowerCase()) {
            return res.status(401).json({
                error: 'Invalid signature. Wallet verification failed.'
            });
        }

        const token = jwt.sign(
            { 
                walletAddress: walletAddress.toLowerCase(),
                role: 'user',
                timestamp: Date.now()
            },
            JWT_SECRET,
            { expiresIn: '24h' }
        );

        res.json({
            success: true,
            token: token,
            walletAddress: walletAddress.toLowerCase(),
            expiresIn: '24h'
        });

    } catch (error) {
        console.error('Wallet authentication error:', error);
        res.status(500).json({
            error: 'Authentication failed: ' + error.message
        });
    }
};

module.exports = {
    authenticateToken,
    authenticateWallet
};