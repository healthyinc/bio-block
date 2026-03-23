# Bio-Block: Secure Document Management System

Bio-Block is a secure, decentralized healthcare document management system for the Web3 era. It leverages blockchain technology, IPFS (InterPlanetary File System), and vector databases to provide secure, verifiable, and privacy-preserving document storage and management for healthcare data.

## Key Features

### Advanced Security & Privacy

- **Streaming Encryption**: Memory-safe encryption for large files (>5MB) with real-time progress tracking
- **PHI Anonymization**: Automatic anonymization of Personal Health Information in Excel and image files
- **Blockchain Verification**: Document hashes stored on Ethereum for tamper-proof verification
- **Decentralized Storage**: IPFS-based storage with encryption and secure access controls

### Healthcare Data Management

- **Multi-format Support**: Excel (.xlsx, .xls), CSV, ODS, TSV, and other spreadsheet formats (.xlsm, .xlsb), plus medical images (.jpg, .jpeg, .png)
- **Smart Anonymization**: Wallet-based hashing for personal data, OCR+NLP for medical images
- **Preview System**: Free 5% preview of Excel data for evaluation before purchase
- **Metadata Collection**: Comprehensive tagging with disease types, demographics, and data sources

### Intelligent Search & Discovery

- **Vector Search**: Natural language queries using ChromaDB for semantic document discovery
- **Advanced Filtering**: Filter by data type, gender, source, file type, and other metadata
- **Combined Search**: Semantic search enhanced with metadata filters for precise results

### Marketplace & Economics

- **Document Marketplace**: Set prices and earn from document sales
- **Earnings Tracking**: Real-time earnings display and withdrawal functionality
- **Preview Downloads**: Free evaluation of data quality before purchase
- **Wallet Integration**: Seamless Ethereum wallet connectivity

## Architecture

Bio-Block follows a microservices architecture with separate frontend and backend services:

![Architecture Diagram](./image.png)

## Project Structure

```
healthy/
├── prototype/                 # React frontend application
│   ├── src/
│   │   ├── App.js            # Main application with navigation
│   │   ├── contractService.js # Smart contract interactions
│   │   ├── upload_data.js    # Document upload interface with streaming encryption
│   │   ├── search_data.js    # Document search interface with smart decryption
│   │   ├── Dashboard.js      # User dashboard with earnings and document management
│   │   ├── encryptionUtils.js # Traditional document encryption utilities
│   │   ├── utils/
│   │   │   └── streamingEncryption.js # Memory-safe streaming encryption for large files
│   │   └── DocumentStorage.sol # Smart contract source
│   └── package.json
├── python_backend/           # FastAPI service
│   ├── main.py               # ChromaDB, search endpoints, and image PHI anonymization
│   ├── requirements.txt
│   ├── vercel.json           # Vercel deployment config
│   ├── tests/                # Python API test suite
│   │   ├── test_api.py       # Comprehensive API tests with unittest
│   │   └── test.jpg          # Test image for anonymization tests
│   └── chroma_db/            # Local ChromaDB storage
├── javascript_backend/        # Express.js API server
│   ├── controllers/          # Business logic controllers
│   │   ├── anonymizeController.js # Excel file anonymization logic
│   │   ├── ipfsController.js      # IPFS interaction logic
│   │   └── healthController.js    # Health check logic
│   ├── routes/              # API route definitions
│   │   ├── anonymize.js     # Excel anonymization routes
│   │   ├── ipfs.js          # IPFS routes
│   │   └── health.js        # Health check routes
│   ├── tests/               # JavaScript API test suite
│   │   ├── api.test.js      # Mocha/Chai/SuperTest API tests
│   │   └── test.xlsx        # Test Excel file with PHI data
│   ├── server.js            # Main server file
│   ├── vercel.json          # Vercel deployment config
│   └── package.json
└── README.md
```

### Frontend (React)

- Modern UI built with React.js and Tailwind CSS
- Interactive progress tracking for uploads and encryption
- Wallet integration for Ethereum connectivity
- Document marketplace and earnings dashboard

### JavaScript Backend (Express.js - Port 3001)

- Excel file processing and PHI anonymization
- IPFS file upload handling
- Preview generation for Excel files
- RESTful API with MVC architecture

### Python Backend (FastAPI - Port 3002)

- Vector database operations using ChromaDB
- Image PHI anonymization using Presidio/OCR
- Semantic search and document filtering
- Advanced ML-based text processing

### Smart Contracts (Solidity)

- Document verification on Ethereum blockchain
- Marketplace functionality for document sales
- Earnings tracking and withdrawal system

## Quick Start

### Prerequisites

- Node.js (v14+)
- uv (v0.10+)
- MetaMask or Ethereum wallet
- Git

### Installation

1. **Clone and setup**

   ```bash
   git clone https://github.com/yourusername/bio-block.git
   cd bio-block
   ```

2. **Backend setup**

   ```bash
   # Python backend
   cd python_backend
   uv sync

   # JavaScript backend
   cd ../javascript_backend
   npm install

   # Frontend
   cd ../prototype
   npm install
   ```

3. **Environment configuration**

   Create `.env` in `prototype/`:

   ```env
   REACT_APP_PINATA_JWT=your_pinata_jwt_key
   REACT_APP_ENCRYPTION_KEY=your_32_byte_encryption_key
   REACT_APP_PYTHON_BACKEND_URL=http://localhost:3002
   REACT_APP_JS_BACKEND_URL=http://localhost:3001
   ```

4. **Run the application**

   ```bash
   # Terminal 1: Python backend
   cd python_backend && uv run uvicorn main:app --reload --port 3002

   # Terminal 2: JavaScript backend
   cd javascript_backend && node server.js

   # Terminal 3: Frontend
   cd prototype && npm start
   ```

Access the application at `http://localhost:3000`

## API Endpoints

### JavaScript Backend (Express.js)

**Local URL**: `http://localhost:3001`

- `GET /` - Root endpoint with API information
- `GET /api/health` - Health check endpoint to verify server status
- `POST /api/anonymize` - Anonymize PHI (Personal Health Information) in spreadsheet files with optional preview generation
  - Input: Spreadsheet file (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb) via multipart form data
  - Optional: Wallet address for personal data anonymization
  - Optional: `generatePreview=true` parameter to create 5% sample preview
  - Output: Full anonymized spreadsheet file, and preview file (if requested) containing first 5% of rows (min 5, max 50)
- `POST /api/ipfs/upload` - Upload a file to IPFS
  - Input: file via multipart form data
  - Output: IPFS hash of the uploaded file
- Organized with MVC architecture (controllers and routes)

### Python Backend (FastAPI)

**Local URL**: `http://localhost:3002`

- `GET /` - Health check and API information
- `POST /store` - Store document summaries and metadata in ChromaDB
- `POST /search` - Search documents using natural language queries
- `POST /filter` - Filter documents by metadata criteria (data type, gender, data source, file type)
- `POST /search_with_filter` - Combined semantic search with metadata filtering
- `POST /anonymize_image` - Anonymize PHI in medical images using Presidio ML models with OCR+spaCy fallback
  - Input: Image file (.jpg, .jpeg, .png) via multipart form data
  - Output: Anonymized image with advanced ML-based PHI redaction
  - Method: Presidio (primary), Tesseract OCR + spaCy NLP (fallback)
- Returns similarity scores, document metadata, and summaries

### Example API Usage

```bash
# Health check - JavaScript backend
curl http://localhost:3001/api/health

# Health check - Python backend
curl http://localhost:3002/

# Search documents (POST request)
curl -X POST http://localhost:3002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "patient information", "k": 5}'

# Filter documents by metadata
curl -X POST http://localhost:3002/filter \
  -H "Content-Type: application/json" \
  -d '{"filters": {"dataType": "Personal", "gender": "Male"}, "n_results": 10}'

# Combined search with filters
curl -X POST http://localhost:3002/search_with_filter \
  -H "Content-Type: application/json" \
  -d '{"query": "diabetes research", "filters": {"dataType": "Institution", "dataSource": "Hospital"}, "n_results": 5}'

# Anonymize medical image using Presidio ML models
curl -X POST http://localhost:3002/anonymize_image \
  -F "file=@medical_scan.jpg"

# Test spreadsheet anonymization (JavaScript backend)
curl -X POST http://localhost:3001/api/anonymize \
  -F "file=@sample_data.xlsx" \
  -F "generatePreview=true"

# Test with different spreadsheet formats
curl -X POST http://localhost:3001/api/anonymize \
  -F "file=@test_sample.csv" \
  -F "generatePreview=true"

curl -X POST http://localhost:3001/api/anonymize \
  -F "file=@test_sample.tsv" \
  -F "generatePreview=true"
```

## Testing

Bio-Block includes comprehensive test suites for both backend services:

### Running Tests

```bash
# Python backend tests (6 tests)
cd python_backend && uv run pytest -q

# JavaScript backend tests (3 tests)
cd javascript_backend && npm test
```

### Test Coverage

- **Python Backend**: Store, search, filter, anonymize image endpoints
- **JavaScript Backend**: Health check, Excel anonymization, IPFS upload
- **Automated Test Data**: Dynamic generation of test files with sample data

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Fork the repository
2. Follow the [Quick Start](#quick-start) guide
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes and test thoroughly
5. Submit a pull request

### Code Style

- Follow existing code patterns
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## Acknowledgements

Built with these amazing technologies:

- [React](https://reactjs.org/) - Frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) - Python backend
- [Express.js](https://expressjs.com/) - JavaScript backend
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [IPFS](https://ipfs.io/) - Decentralized storage
- [Ethereum](https://ethereum.org/) - Blockchain platform
