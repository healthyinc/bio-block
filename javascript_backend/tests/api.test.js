const request = require('supertest');
const expect = require('chai').expect;
const path = require('path');
const app = require('../server');

const testFilePath = path.join(__dirname, 'test.xlsx');

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
    const res = await request(app)
      .post('/api/anonymize')
      .attach('file', testFilePath)
      .field('generatePreview', 'true');
    expect(res.status).to.equal(200);
  });
});
