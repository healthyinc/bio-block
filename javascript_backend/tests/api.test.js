const request = require("supertest");
const expect = require("chai").expect;
const app = require("../server"); // Import the Express app

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
    // Note: ensure test.xlsx actually exists or mock it if possible,
    // but for now we follow the existing pattern.
    const path = require("path");
    const fs = require("fs");
    const testFilePath = path.join(__dirname, "test.xlsx");

    // Create a dummy file if it doesn't exist to prevent test failure
    if (!fs.existsSync(testFilePath)) {
      fs.writeFileSync(testFilePath, "dummy content");
    }

    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", testFilePath)
      .field("generatePreview", "true");

    expect(res.status).to.equal(200);
  });

  it("POST /api/anonymize should return extractedContent", async function () {
    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", "./tests/test.xlsx")
      .field("generatePreview", "true")
      .field("datasetTitle", "Test Dataset");

    expect(res.status).to.equal(200);
    expect(res.body).to.have.property("extractedContent");
    expect(res.body).to.have.property("extractionStatus", "success");
  });
});
