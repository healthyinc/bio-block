const request = require('supertest');
const expect = require('chai').expect;
const path = require('path');

// Use the exported Express app so tests run without needing a separately started server
const app = require('../server');

describe('API Endpoints', function() {
  it('GET / should return API info', async function() {
    const res = await request(app).get('/');
    expect(res.status).to.equal(200);
    expect(res.body).to.have.property('message');
  });

  it('GET /api/health should return health status', async function() {
    const res = await request(app).get('/api/health');
    expect(res.status).to.equal(200);
  });

  it('POST /api/anonymize should anonymize Excel file', async function() {
    // Resolve the test file path relative to this test file
    const testFilePath = path.join(__dirname, 'test.xlsx');
    const res = await request(app)
      .post('/api/anonymize')
      .attach('file', testFilePath)
      .field('generatePreview', 'true');
    // Helpful logs for troubleshooting in CI
    console.log('Response status:', res.status);
    console.log('Response body:', res.body);
    expect(res.status).to.equal(200);
  });
});
