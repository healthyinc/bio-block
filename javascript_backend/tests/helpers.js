/**
 * Test Helpers and Mock Data
 * Shared utilities and mock data for test suite
 */

// Mock IPFS response
const mockIPFSResponse = {
  IpfsHash: 'QmTest123456789abcdefghijk',
  PinSize: 1234,
  Timestamp: '2026-02-16T10:00:00.000Z'
};

// Mock Pinata API success response
const mockPinataSuccess = {
  data: mockIPFSResponse
};

// Mock anonymized data
const mockAnonymizedData = [
  { WID: 'W001', Age: 45, Gender: 'M', Diagnosis: 'Condition A' },
  { WID: 'W002', Age: 32, Gender: 'F', Diagnosis: 'Condition B' },
  { WID: 'W003', Age: 58, Gender: 'M', Diagnosis: 'Condition C' }
];

// Mock Excel data with PHI
const mockExcelDataWithPHI = [
  { Name: 'John Doe', Age: 45, SSN: '123-45-6789', Email: 'john@example.com' },
  { Name: 'Jane Smith', Age: 32, SSN: '987-65-4321', Email: 'jane@example.com' }
];

// Mock file upload data
const mockFileUpload = {
  originalname: 'test-data.xlsx',
  mimetype: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  size: 5000,
  buffer: Buffer.from('mock file content')
};

// Mock Qdrant search results
const mockQdrantResults = [
  {
    id: 'doc1',
    score: 0.95,
    payload: {
      content: 'Medical record content',
      metadata: { date: '2026-01-15' }
    }
  },
  {
    id: 'doc2',
    score: 0.88,
    payload: {
      content: 'Another medical record',
      metadata: { date: '2026-01-20' }
    }
  }
];

// Mock environment variables
const mockEnv = {
  PINATA_API_KEY: 'test-api-key',
  PINATA_SECRET_KEY: 'test-secret-key',
  QDRANT_URL: 'http://localhost:6333',
  PORT: 3001
};

// Helper function to create mock request
function createMockRequest(options = {}) {
  return {
    body: options.body || {},
    query: options.query || {},
    params: options.params || {},
    file: options.file || null,
    headers: options.headers || {}
  };
}

// Helper function to create mock response
function createMockResponse() {
  const res = {};
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (data) => {
    res.body = data;
    return res;
  };
  res.send = (data) => {
    res.body = data;
    return res;
  };
  return res;
}

// Helper to detect PHI patterns
const phiPatterns = {
  name: /\b[A-Z][a-z]+\s[A-Z][a-z]+\b/,
  ssn: /\b\d{3}-\d{2}-\d{4}\b/,
  email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/,
  phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/,
  address: /\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b/i
};

module.exports = {
  mockIPFSResponse,
  mockPinataSuccess,
  mockAnonymizedData,
  mockExcelDataWithPHI,
  mockFileUpload,
  mockQdrantResults,
  mockEnv,
  createMockRequest,
  createMockResponse,
  phiPatterns
};
