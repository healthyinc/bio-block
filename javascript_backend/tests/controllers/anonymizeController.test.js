const request = require("supertest");
const { expect } = require("chai");
const XLSX = require("xlsx");
const app = require("../../server");

/**
 * Helper: create an in-memory XLSX buffer from an array-of-arrays.
 * Optionally accepts a sheet name (defaults to "Sheet1").
 */
function createXlsxBuffer(data, sheetName = "Sheet1") {
  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.aoa_to_sheet(data);
  XLSX.utils.book_append_sheet(wb, ws, sheetName);
  return XLSX.write(wb, { bookType: "xlsx", type: "buffer" });
}

/**
 * Helper: create an XLSX buffer with multiple sheets.
 * sheets is an array of { name, data } objects.
 */
function createMultiSheetXlsxBuffer(sheets) {
  const wb = XLSX.utils.book_new();
  sheets.forEach(({ name, data }) => {
    const ws = XLSX.utils.aoa_to_sheet(data);
    XLSX.utils.book_append_sheet(wb, ws, name);
  });
  return XLSX.write(wb, { bookType: "xlsx", type: "buffer" });
}

/**
 * Helper: send a file to the anonymize endpoint and parse the result.
 * Returns the parsed output as an array-of-arrays for the first sheet.
 */
async function anonymizeAndParse(buffer, filename = "test.xlsx") {
  const res = await request(app)
    .post("/api/anonymize")
    .attach("file", buffer, filename)
    .field("generatePreview", "true");

  expect(res.status).to.equal(200);
  expect(res.body.success).to.be.true;

  const anonymizedBuffer = Buffer.from(res.body.files.main.data, "base64");
  const wb = XLSX.read(anonymizedBuffer, { type: "buffer" });
  const ws = wb.Sheets[wb.SheetNames[0]];
  return XLSX.utils.sheet_to_json(ws, { header: 1 });
}

const WID_PATTERN = /^WID_[a-f0-9]{8}$/;

describe("PHI Detection in Anonymization", function () {
  this.timeout(15000);

  // ------------------------------------------------------------------
  // 1. Standard PHI keyword detection
  // ------------------------------------------------------------------
  describe("Standard PHI column detection", function () {
    const phiHeaders = [
      "DOB",
      "Date of Birth",
      "Address",
      "Phone",
      "Mobile",
      "Email",
      "SSN",
      "Social Security",
      "MRN",
      "Medical Record",
      "Health Plan",
      "License",
      "Account Number",
      "IP Address",
      "Device ID",
      "Biometric",
      "Photo",
      "Facial",
      "Fingerprint",
      "Signature",
      "First Name",
      "Last Name",
      "Name",
    ];

    phiHeaders.forEach((header) => {
      it(`should anonymize column "${header}"`, async function () {
        const data = [
          [header, "Score"],
          ["Sensitive Value", 85],
          ["Another Sensitive", 90],
        ];
        const buffer = createXlsxBuffer(data);
        const output = await anonymizeAndParse(buffer);

        for (let i = 1; i < output.length; i++) {
          expect(String(output[i][0])).to.match(
            WID_PATTERN,
            `Column "${header}" row ${i} should be anonymized`
          );
          // Non-PHI column preserved
          expect(output[i][1]).to.be.a("number");
        }
      });
    });
  });

  // ------------------------------------------------------------------
  // 2. Case-insensitive detection
  // ------------------------------------------------------------------
  describe("Case-insensitive detection", function () {
    ["email", "EMAIL", "Email", "eMaIl"].forEach((variant) => {
      it(`should detect "${variant}" regardless of case`, async function () {
        const data = [
          [variant, "Score"],
          ["test@example.com", 95],
        ];
        const buffer = createXlsxBuffer(data);
        const output = await anonymizeAndParse(buffer);

        expect(String(output[1][0])).to.match(WID_PATTERN);
        expect(output[1][1]).to.equal(95);
      });
    });
  });

  // ------------------------------------------------------------------
  // 3. Partial / composite header match
  // ------------------------------------------------------------------
  describe("Partial match detection", function () {
    const compositeHeaders = [
      "patient_dob",
      "user_email",
      "home_address",
      "cell_phone",
      "patient_name",
      "emergency_phone_number",
    ];

    compositeHeaders.forEach((header) => {
      it(`should detect PHI in composite header "${header}"`, async function () {
        const data = [
          [header, "safe_column"],
          ["sensitive", "safe"],
        ];
        const buffer = createXlsxBuffer(data);
        const output = await anonymizeAndParse(buffer);

        expect(String(output[1][0])).to.match(WID_PATTERN, `"${header}" should be detected as PHI`);
        expect(output[1][1]).to.equal("safe");
      });
    });
  });

  // ------------------------------------------------------------------
  // 4. Non-PHI columns must be preserved
  // ------------------------------------------------------------------
  describe("Non-PHI columns are preserved", function () {
    const safeHeaders = [
      "age",
      "gender",
      "diagnosis",
      "blood_type",
      "department",
      "lab_results",
      "medication",
      "hospital_code",
      "ward_number",
      "treatment_duration",
    ];

    safeHeaders.forEach((header) => {
      it(`should NOT anonymize non-PHI column "${header}"`, async function () {
        const data = [[header], ["original_value_123"], ["another_value_456"]];
        const buffer = createXlsxBuffer(data);
        const output = await anonymizeAndParse(buffer);

        for (let i = 1; i < output.length; i++) {
          expect(String(output[i][0])).to.not.match(
            WID_PATTERN,
            `"${header}" should NOT be detected as PHI`
          );
        }
      });
    });
  });

  // ------------------------------------------------------------------
  // 5. Patient ID column detection & consistent hashing
  // ------------------------------------------------------------------
  describe("Patient ID column detection", function () {
    it("should detect 'Patient ID' as identifier column", async function () {
      const data = [
        ["Patient ID", "Name", "Score"],
        ["PAT001", "John Doe", 85],
        ["PAT002", "Jane Smith", 90],
      ];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      // Patient ID column: anonymized
      expect(String(output[1][0])).to.match(WID_PATTERN);
      // Name column: PHI keyword, also anonymized
      expect(String(output[1][1])).to.match(WID_PATTERN);
      // Score: non-PHI, preserved
      expect(output[1][2]).to.equal(85);
    });

    it("should produce consistent WIDs for the same patient ID", async function () {
      const data = [
        ["Patient ID", "Name", "Visit"],
        ["PAT001", "John Doe", "2024-01-01"],
        ["PAT001", "John Doe", "2024-02-15"],
        ["PAT002", "Jane Smith", "2024-01-10"],
      ];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      // Same patient ID → same WID
      expect(output[1][0]).to.equal(output[2][0]);
      // Different patient ID → different WID
      expect(output[1][0]).to.not.equal(output[3][0]);
    });

    it("should detect case variants of Patient ID header", async function () {
      const variants = ["patient_id", "PATIENT ID", "Patient Id"];

      for (const header of variants) {
        const data = [
          [header, "Score"],
          ["PAT001", 85],
        ];
        const buffer = createXlsxBuffer(data);
        const output = await anonymizeAndParse(buffer);

        expect(String(output[1][0])).to.match(
          WID_PATTERN,
          `"${header}" should be detected as patient ID column`
        );
      }
    });
  });

  // ------------------------------------------------------------------
  // 6. WID format validation
  // ------------------------------------------------------------------
  describe("WID format", function () {
    it("should produce WIDs matching WID_XXXXXXXX (8 hex chars)", async function () {
      const data = [["Name"], ["John Doe"], ["Jane Smith"], ["Bob Wilson"]];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      for (let i = 1; i < output.length; i++) {
        expect(String(output[i][0])).to.match(WID_PATTERN, `Row ${i}: invalid WID format`);
      }
    });
  });

  // ------------------------------------------------------------------
  // 7. Whitespace-normalized matching
  //    The controller strips whitespace from both header and keyword
  //    before comparison, so "firstname" matches "first name".
  // ------------------------------------------------------------------
  describe("Whitespace-normalized matching", function () {
    it('should detect "firstname" (no space) as PHI', async function () {
      const data = [
        ["firstname", "score"],
        ["John", 85],
      ];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      expect(String(output[1][0])).to.match(WID_PATTERN);
      expect(output[1][1]).to.equal(85);
    });

    it('should detect "socialsecurity" (no space) as PHI', async function () {
      const data = [
        ["socialsecurity", "age"],
        ["123-45-6789", 30],
      ];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      expect(String(output[1][0])).to.match(WID_PATTERN);
      expect(output[1][1]).to.equal(30);
    });

    it('should detect "medicalrecord" (no space) as PHI', async function () {
      const data = [
        ["medicalrecord", "status"],
        ["MR-12345", "active"],
      ];
      const buffer = createXlsxBuffer(data);
      const output = await anonymizeAndParse(buffer);

      expect(String(output[1][0])).to.match(WID_PATTERN);
      expect(output[1][1]).to.equal("active");
    });
  });

  // ------------------------------------------------------------------
  // 8. Multi-sheet handling
  // ------------------------------------------------------------------
  describe("Multi-sheet PHI detection", function () {
    it("should anonymize PHI columns across multiple sheets", async function () {
      const buffer = createMultiSheetXlsxBuffer([
        {
          name: "Demographics",
          data: [
            ["Name", "Age"],
            ["John Doe", 45],
          ],
        },
        {
          name: "Contact",
          data: [
            ["Email", "Department"],
            ["john@example.com", "Cardiology"],
          ],
        },
      ]);

      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", buffer, "multi.xlsx")
        .field("generatePreview", "true");

      expect(res.status).to.equal(200);

      const anonymizedBuffer = Buffer.from(res.body.files.main.data, "base64");
      const wb = XLSX.read(anonymizedBuffer, { type: "buffer" });

      // Sheet 1: "Name" anonymized, "Age" preserved
      const sheet1 = XLSX.utils.sheet_to_json(wb.Sheets["Demographics"], {
        header: 1,
      });
      expect(String(sheet1[1][0])).to.match(WID_PATTERN);
      expect(sheet1[1][1]).to.equal(45);

      // Sheet 2: "Email" anonymized, "Department" preserved
      const sheet2 = XLSX.utils.sheet_to_json(wb.Sheets["Contact"], {
        header: 1,
      });
      expect(String(sheet2[1][0])).to.match(WID_PATTERN);
      expect(sheet2[1][1]).to.equal("Cardiology");
    });
  });

  // ------------------------------------------------------------------
  // 9. Edge cases
  // ------------------------------------------------------------------
  describe("Edge cases", function () {
    it("should handle spreadsheet with headers only", async function () {
      const data = [["Name", "Email", "Score"]];
      const buffer = createXlsxBuffer(data);

      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", buffer, "headers_only.xlsx")
        .field("generatePreview", "true");

      expect(res.status).to.equal(200);
    });

    it("should handle numeric column headers without error", async function () {
      const data = [
        [123, 456, 789],
        ["value1", "value2", "value3"],
      ];
      const buffer = createXlsxBuffer(data);

      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", buffer, "numeric_headers.xlsx")
        .field("generatePreview", "true");

      expect(res.status).to.equal(200);

      const output = await anonymizeAndParse(buffer, "numeric_headers.xlsx");

      // Numeric headers should not match any PHI keyword → values preserved
      expect(output[1][0]).to.equal("value1");
    });

    it("should handle null/undefined cells in PHI columns gracefully", async function () {
      const data = [
        ["Name", "Score"],
        ["John Doe", 85],
        [null, 90],
        [undefined, 75],
        ["", 60],
      ];
      const buffer = createXlsxBuffer(data);

      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", buffer, "sparse.xlsx")
        .field("generatePreview", "true");

      expect(res.status).to.equal(200);
    });

    it("should reject non-spreadsheet files", async function () {
      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", Buffer.from("not a spreadsheet"), "test.txt");

      expect(res.status).to.be.oneOf([400, 500]);
    });

    it("should return 400 when no file is uploaded", async function () {
      const res = await request(app).post("/api/anonymize");

      expect(res.status).to.equal(400);
      expect(res.body).to.have.property("error");
    });
  });

  // ------------------------------------------------------------------
  // 10. CSV format support
  // ------------------------------------------------------------------
  describe("CSV format support", function () {
    it("should detect PHI columns in .csv files", async function () {
      const csvContent = "Name,Age,Email\nJohn Doe,45,john@test.com\n";
      const buffer = Buffer.from(csvContent);

      const res = await request(app)
        .post("/api/anonymize")
        .attach("file", buffer, "test.csv")
        .field("generatePreview", "true");

      expect(res.status).to.equal(200);
    });
  });
});
