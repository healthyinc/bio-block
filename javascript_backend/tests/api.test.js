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
});
