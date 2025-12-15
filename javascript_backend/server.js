require('dotenv').config();
const express = require('express');
const cors = require('cors');

// Import Routes
const anonymizeRoutes = require('./routes/anonymize');
const healthRoutes = require('./routes/health');
const ipfsRoutes = require('./routes/ipfs');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/anonymize', anonymizeRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/ipfs', ipfsRoutes);

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Bio-Block JavaScript Backend API',
        status: 'OK',
        timestamp: new Date().toISOString(),
        endpoints: [
            '/api/health',
            '/api/anonymize',
            '/api/ipfs/upload'
        ]
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    res.status(500).json({ 
        error: 'Internal server error' 
    });
});

// Start server (for local development)
if (require.main === module) {
    app.listen(PORT, () => {
        // Server started successfully
    });
}

// Export for Vercel
module.exports = app;