const request = require("supertest");
const expect = require("chai").expect;
const XLSX = require("xlsx");
const path = require("path");
const fs = require("fs");
const app = require("../server");

const testFilePath = path.join(__dirname, "test.xlsx");

before(function () {
  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.aoa_to_sheet([
    ["Patient ID", "Name", "DOB"],
    ["P001", "John Doe", "1990-01-01"],
    ["P002", "Jane Smith", "1985-05-15"],
  ]);
  XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  XLSX.writeFile(wb, testFilePath);
});

after(function () {
  if (fs.existsSync(testFilePath)) {
    fs.unlinkSync(testFilePath);
  }
});

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
    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", testFilePath)
      .field("generatePreview", "true");

    expect(res.status).to.equal(200);
  });

  it("POST /api/quality/profile should return dataset quality profile", async function () {
    const res = await request(app).post("/api/quality/profile").attach("file", testFilePath);
    expect(res.status).to.equal(200);
    expect(res.body).to.have.property("success", true);
    expect(res.body).to.have.property("report");
  });

  it("POST /api/anonymize with preview should return main and preview files", async function () {
    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", testFilePath)
      .field("generatePreview", "true");

    expect(res.status).to.equal(200);
    expect(res.body).to.have.property("success", true);
    expect(res.body).to.have.nested.property("files.main.data");
    expect(res.body).to.have.nested.property("files.main.filename");
    expect(res.body).to.have.nested.property("files.preview.data");
    expect(res.body).to.have.nested.property("files.preview.filename");
  });

  it("POST /api/anonymize should redact PHI patterns in cell content", async function () {
    // Create an Excel file with PHI in cell content but non-PHI column headers
    const cellPhiPath = path.join(__dirname, "test_cell_phi.xlsx");
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet([
      ["Notes", "Details", "Record"],
      ["Contact at 555-123-4567", "SSN: 123-45-6789", "john@hospital.com"],
      ["Normal text here", "No PHI data", "Just notes"],
    ]);
    XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
    XLSX.writeFile(wb, cellPhiPath);

    const res = await request(app)
      .post("/api/anonymize")
      .attach("file", cellPhiPath)
      .field("generatePreview", "true");

    expect(res.status).to.equal(200);

    // Parse the anonymized output and verify PHI was redacted
    const outputBuffer = Buffer.from(res.body.files.main.data, "base64");
    const outputWb = XLSX.read(outputBuffer, { type: "buffer" });
    const outputData = XLSX.utils.sheet_to_json(outputWb.Sheets["Sheet1"], { header: 1 });

    // Row 1 (index 1) should have PHI redacted from cells
    expect(outputData[1][0]).to.equal("[PHI_REDACTED]"); // phone number
    expect(outputData[1][1]).to.equal("[PHI_REDACTED]"); // SSN
    expect(outputData[1][2]).to.equal("[PHI_REDACTED]"); // email

    // Row 2 (index 2) should remain unchanged (no PHI)
    expect(outputData[2][0]).to.equal("Normal text here");
    expect(outputData[2][1]).to.equal("No PHI data");
    expect(outputData[2][2]).to.equal("Just notes");

    fs.unlinkSync(cellPhiPath);
  });

  it("POST /api/quality/profile should return statistical summaries for numeric columns", async function () {
    const statsFilePath = path.join(__dirname, "test_stats.xlsx");
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet([
      ["Name", "Score"],
      ["Alice", 10],
      ["Bob", 20],
      ["Charlie", 30],
      ["David", 40],
      ["Eve", 50],
    ]);
    XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
    XLSX.writeFile(wb, statsFilePath);

    const res = await request(app).post("/api/quality/profile").attach("file", statsFilePath);

    expect(res.status).to.equal(200);
    const scoreCol = res.body.report.sheets[0].columns.find((c) => c.name === "Score");

    expect(scoreCol).to.have.property("inferredType", "number");
    expect(scoreCol).to.have.property("statistics");
    expect(scoreCol.statistics).to.deep.include({
      min: 10,
      max: 50,
      mean: 30,
      median: 30,
      count: 5,
    });
    expect(scoreCol.statistics).to.have.property("stdDev");
    expect(scoreCol.statistics.stdDev).to.be.closeTo(14.1421, 0.0001);

    fs.unlinkSync(statsFilePath);
  });
});
