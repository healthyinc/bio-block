/**
 * Test Setup
 * Global configuration for test suite
 */

// Set test environment
process.env.NODE_ENV = "test";

// Mock environment variables for tests
process.env.PINATA_API_KEY = "test-api-key";
process.env.PINATA_SECRET_KEY = "test-secret-key";
process.env.QDRANT_URL = "http://localhost:6333";
process.env.PORT = 3001;

// Suppress console.log during tests (optional)
if (process.env.SUPPRESS_LOGS === "true") {
  global.console = {
    ...console,
    log: () => {},
    info: () => {},
    debug: () => {},
  };
}

module.exports = {};
