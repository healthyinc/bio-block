const request = require('supertest');
const expect = require('chai').expect;
const app = require('../server');

describe('Anonymize Upload Limits', function () {
  it('accepts a small valid CSV upload', async function () {
    const smallCsv = Buffer.from('patient_id,name\n1,Alice\n2,Bob\n', 'utf8');

    const res = await request(app)
      .post('/api/anonymize')
      .attach('file', smallCsv, { filename: 'small.csv', contentType: 'text/csv' })
      .field('generatePreview', 'true');

    expect(res.status).to.equal(200);
    expect(res.body).to.have.property('success', true);
  });

  it('rejects upload larger than configured limit', async function () {
    const overLimitBuffer = Buffer.alloc(20 * 1024 * 1024 + 1024, 'a');

    const res = await request(app)
      .post('/api/anonymize')
      .attach('file', overLimitBuffer, { filename: 'large.csv', contentType: 'text/csv' });

    expect(res.status).to.not.equal(200);
  });
});
