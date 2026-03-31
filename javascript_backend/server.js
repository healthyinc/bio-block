require('dotenv').config();
const express = require('express');
const cors = require('cors');

// Import Routes
const anonymizeRoutes = require('./routes/anonymize');
const healthRoutes = require('./routes/health');
const ipfsRoutes = require('./routes/ipfs');
const encryptionRoutes = require('./routes/encryption');

// Import Authentication
const { authenticateToken, authenticateWallet } = require('./middleware/auth');

const app = express();
const PORT = process.env.PORT || 3001;

// CORS Configuration - Security fix: Restrict origins
const allowedOrigins = [
    'http://localhost:3000',
    'https://bio-block.vercel.app',
    process.env.FRONTEND_URL
].filter(Boolean);

app.use(cors({
    origin: function (origin, callback) {
        if (!origin) return callback(null, true);
        
        if (allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// Rate limiting middleware
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: {
        error: 'Too many requests from this IP, please try again later.'
    }
});

app.use(limiter);

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Authentication endpoint (no auth required)
app.post('/api/auth/wallet', authenticateWallet);

// Protected Routes (require authentication)
app.use('/api/anonymize', authenticateToken, anonymizeRoutes);
app.use('/api/ipfs', authenticateToken, ipfsRoutes);
app.use('/api', encryptionRoutes);

// Public Routes
app.use('/api/health', healthRoutes);

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Bio-Block JavaScript Backend API - Secure Version',
        status: 'OK',
        timestamp: new Date().toISOString(),
        security: {
            authentication: 'JWT with wallet signature verification',
            encryption: 'Server-side AES-256-GCM with PBKDF2',
            cors: 'Restricted to allowed origins',
            rateLimit: '100 requests per 15 minutes'
        },
        endpoints: [
            'POST /api/auth/wallet - Authenticate with wallet signature',
            'GET /api/health - Health check (public)',
            'POST /api/anonymize - Anonymize files (protected)',
            'POST /api/ipfs/upload - Upload to IPFS (protected)',
            'POST /api/encrypt - Encrypt files (protected)',
            'POST /api/decrypt - Decrypt files (protected)',
            'POST /api/encrypt-stream - Streaming encryption (protected)',
            'POST /api/decrypt-stream - Streaming decryption (protected)'
        ]
    });
});

// Security headers middleware
app.use((req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    next();
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    
    const isDevelopment = process.env.NODE_ENV === 'development';
    
    res.status(500).json({ 
        error: 'Internal server error',
        ...(isDevelopment && { details: error.message })
    });
});

// Start server (for local development)
if (require.main === module) {
    app.listen(PORT, () => {
        console.log(`Secure Bio-Block Backend running on http://localhost:${PORT}`);
        console.log(`Security features enabled:`);
        console.log(`   - JWT Authentication with wallet verification`);
        console.log(`   - Server-side encryption (AES-256-GCM)`);
        console.log(`   - CORS restricted to allowed origins`);
        console.log(`   - Rate limiting (100 req/15min)`);
        console.log(`   - Security headers enabled`);
        console.log(`Endpoints:`);
        console.log(`   - Health check: http://localhost:${PORT}/api/health`);
        console.log(`   - Authentication: http://localhost:${PORT}/api/auth/wallet`);
    });
}

// Export for Vercel
module.exports = app;