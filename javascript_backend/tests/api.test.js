const request = require("supertest");
const expect = require("chai").expect;

// Create a proper supertest instance pointing to the server URL
const app = "http://localhost:3001";

describe("API Endpoints", function () {
  it("GET / should return API info", async function () {
    const res = await request(app).get("/");
    expect(res.status).to.equal(200);
    expect(res.body).to.have.property("message");
  });

  it("GET /api/health should return health status", async function () {
    const res = await request(app).get("/api/health");
    expect(res.status).to.equal(200);
  });

  it("POST /api/anonymize should anonymize Excel file", async function () {
    // Using the test.xlsx file we created in the tests directory
    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", "./tests/test.xlsx")
      .field("generatePreview", "true");
    console.log("Response status:", res.status);
    console.log("Response body:", res.body);
    expect(res.status).to.equal(200);
  });

   it('POST /api/anonymize should return extractedContent', async function() {
    const res = await request(app)
      .post('/api/anonymize')
      .attach('file', './tests/test.xlsx')
      .field('generatePreview', 'true')
      .field('datasetTitle', 'Test Dataset');
    
    expect(res.status).to.equal(200);
    expect(res.body).to.have.property('extractedContent');
    expect(res.body).to.have.property('extractionStatus', 'success');
  });
});
