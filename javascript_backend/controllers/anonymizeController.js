const multer = require("multer");
const XLSX = require("xlsx");
const crypto = require("crypto");
const { v4: uuidv4 } = require("uuid");

const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    // Accept Excel files, CSV, ODS, TSV and other spreadsheet formats
    const allowedMimeTypes = [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
      "application/vnd.ms-excel", // .xls
      "text/csv", // .csv
      "application/csv", // .csv (alternative)
      "application/vnd.oasis.opendocument.spreadsheet", // .ods
      "text/tab-separated-values", // .tsv
      "application/vnd.ms-excel.sheet.macroEnabled.12", // .xlsm
      "application/vnd.ms-excel.sheet.binary.macroEnabled.12", // .xlsb
    ];

    // Also check file extension as a fallback
    const allowedExtensions = /\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i;

    if (allowedMimeTypes.includes(file.mimetype) || allowedExtensions.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(
        new Error(
          "Only spreadsheet files (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb) are allowed"
        ),
        false
      );
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024 * 1024, // 10GB limit
  },
});

const phiKeywords = [
  "dob",
  "date of birth",
  "address",
  "phone",
  "mobile",
  "email",
  "ssn",
  "social security",
  "mrn",
  "medical record",
  "health plan",
  "license",
  "account number",
  "ip address",
  "device id",
  "biometric",
  "photo",
  "facial",
  "fingerprint",
  "signature",
  "first name",
  "last name",
  "name",
];

const phiKeywordsLower = phiKeywords.map((keyword) => keyword.toLowerCase());
const phiKeywordsNormalized = phiKeywordsLower.map((keyword) => keyword.replace(/\s+/g, ""));

function normalizeHeaderValue(value) {
  return value.toLowerCase().replace(/\s+/g, "");
}

function generateAnonymizedId(input, index) {
  const hash = crypto.createHash("sha256").update(input).digest("hex");
  const shortHash = hash.substring(0, 8);
  return `WID_${shortHash}`;
}

function generateUUID() {
  return uuidv4();
}

const anonymizeFile = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded. Please upload a spreadsheet file.",
      });
    }

    const walletAddress = req.body.walletAddress;
    const generatePreview = req.body.generatePreview === "true";
    const isPersonalData = !!walletAddress;

    console.log("Processing anonymization:", {
      isPersonalData,
      generatePreview,
      fileName: req.file.originalname,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      walletAddress: walletAddress ? `${walletAddress.substring(0, 6)}...` : "N/A",
    });

    // Parse the file using XLSX library which supports multiple formats
    let workbook;
    try {
      // XLSX library automatically detects format based on file content
      workbook = XLSX.read(req.file.buffer, {
        type: "buffer",
        cellDates: true,
        cellNF: false,
        cellText: false,
      });
    } catch (parseError) {
      console.error("File parsing error:", parseError);
      return res.status(400).json({
        error:
          "Failed to parse spreadsheet file. Please ensure the file is not corrupted and is in a supported format.",
      });
    }
    const sheetColumnRefs = {};
    const sheetDataByName = {};
    const startTime = Date.now();

    workbook.SheetNames.forEach((sheetName) => {
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1, blankrows: false });
      sheetDataByName[sheetName] = jsonData;

      if (jsonData.length === 0) {
      sheetColumnRefs[sheetName] = {};
      return;
    }

    const headers = jsonData[0] || [];
    let patientIdCol = null;

    for (let index = 0; index < headers.length; index++) {
     const header = headers[index];
     if (header && typeof header === "string") {
      const headerLower = header.toLowerCase();
      if (headerLower.includes("patient") && headerLower.includes("id")) {
        patientIdCol = index;
        break;
      }
    }
  }

  if (patientIdCol !== null) {
    sheetColumnRefs[sheetName] = { patientIdCol };
  } else {
    sheetColumnRefs[sheetName] = { useUUID: true };
  }
});

const cleanedWorkbook = XLSX.utils.book_new();
const cleanedSheetData = {};

    workbook.SheetNames.forEach((sheetName) => {
  const jsonData = sheetDataByName[sheetName] || [];

  if (jsonData.length === 0) {
    const worksheet = workbook.Sheets[sheetName];
    XLSX.utils.book_append_sheet(cleanedWorkbook, worksheet, sheetName);
    cleanedSheetData[sheetName] = jsonData;
    return;
  }

  const headers = jsonData[0] || [];
  const cleanedData = jsonData;
  cleanedSheetData[sheetName] = cleanedData;

  const sheetRefs = sheetColumnRefs[sheetName] || {};
  const columnsToMask = [];

  for (let index = 0; index < headers.length; index++) {
    const header = headers[index];
    if (header && typeof header === "string") {
      const headerLower = header.toLowerCase();
      const headerNormalized = normalizeHeaderValue(headerLower);
      const isPatientIdColumn = index === sheetRefs.patientIdCol;

      let isPhiColumn = false;
      for (let k = 0; k < phiKeywordsLower.length; k++) {
        if (
          headerLower.includes(phiKeywordsLower[k]) ||
          headerNormalized.includes(phiKeywordsNormalized[k])
        ) {
          isPhiColumn = true;
          break;
        }
      }

      if (isPatientIdColumn || isPhiColumn) {
        columnsToMask.push(index);
      }
    }
  }

  if (sheetRefs.useUUID) {
    const sharedPersonalId =
      isPersonalData && walletAddress ? generateAnonymizedId(walletAddress, 1) : null;

    for (let i = 1; i < cleanedData.length; i++) {
      const row = cleanedData[i];
      if (row && row.some((cell) => cell !== undefined && cell !== null && cell !== "")) {
        const id = sharedPersonalId || generateAnonymizedId(generateUUID(), i);

        for (let c = 0; c < columnsToMask.length; c++) {
          const colIndex = columnsToMask[c];
          if (row[colIndex] !== undefined) {
            row[colIndex] = id;
          }
        }
      }
    }
  } else {
    for (let i = 1; i < cleanedData.length; i++) {
      const row = cleanedData[i];

      if (sheetRefs.patientIdCol !== undefined && row && row[sheetRefs.patientIdCol]) {
        const patientId = String(row[sheetRefs.patientIdCol]).toLowerCase().trim();
        const id = generateAnonymizedId(patientId, i);

        for (let c = 0; c < columnsToMask.length; c++) {
          const colIndex = columnsToMask[c];
          if (row[colIndex] !== undefined) {
            row[colIndex] = id;
          }
        }
      }
    }
  }

  const newWorksheet = XLSX.utils.aoa_to_sheet(cleanedData);
  XLSX.utils.book_append_sheet(cleanedWorkbook, newWorksheet, sheetName);
});

    // Determine output format based on input file extension
    const originalFileName = req.file.originalname;
    const fileExtension = originalFileName.split(".").pop().toLowerCase();

    let outputFormat = "xlsx"; // Default format
    let outputMimeType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

    // Map file extensions to XLSX library book types
    switch (fileExtension) {
      case "xls":
        outputFormat = "xls";
        outputMimeType = "application/vnd.ms-excel";
        break;
      case "csv":
        outputFormat = "csv";
        outputMimeType = "text/csv";
        break;
      case "ods":
        outputFormat = "ods";
        outputMimeType = "application/vnd.oasis.opendocument.spreadsheet";
        break;
      case "tsv":
        outputFormat = "txt"; // XLSX library uses 'txt' for TSV
        outputMimeType = "text/tab-separated-values";
        break;
      case "xlsm":
        outputFormat = "xlsm";
        outputMimeType = "application/vnd.ms-excel.sheet.macroEnabled.12";
        break;
      case "xlsb":
        outputFormat = "xlsb";
        outputMimeType = "application/vnd.ms-excel.sheet.binary.macroEnabled.12";
        break;
      default:
        outputFormat = "xlsx";
        outputMimeType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
    }

    const outputBuffer = XLSX.write(cleanedWorkbook, {
      bookType: outputFormat,
      type: "buffer",
    });

             
    // Generate preview if requested
        if (generatePreview) {
      const previewWorkbook = XLSX.utils.book_new();

      cleanedWorkbook.SheetNames.forEach((sheetName) => {
        const jsonData = cleanedSheetData[sheetName] || [];

        if (jsonData.length === 0) {
          const worksheet = cleanedWorkbook.Sheets[sheetName];
          XLSX.utils.book_append_sheet(previewWorkbook, worksheet, sheetName);
          return;
        }

        const totalRows = jsonData.length;
        const previewRows = Math.max(5, Math.min(50, Math.ceil(totalRows * 0.05)));
        const previewData = jsonData.slice(0, previewRows);

        const previewWorksheet = XLSX.utils.aoa_to_sheet(previewData);
        XLSX.utils.book_append_sheet(previewWorkbook, previewWorksheet, sheetName);
      });

      const previewBuffer = XLSX.write(previewWorkbook, {
        bookType: outputFormat,
        type: "buffer",
      });

      const extractedContent = await extractFileContent(
        cleanedWorkbook,
        req.file.originalname,
        req.body.datasetTitle || ""
      );

      console.log(`✅ Anonymization completed in ${Date.now() - startTime}ms`);

      return res.json({
        success: true,
        message: "File anonymized successfully with preview",
        files: {
          main: {
            data: outputBuffer.toString("base64"),
            filename: `anonymized_${originalFileName}`,
            contentType: outputMimeType,
          },
          preview: {
            data: previewBuffer.toString("base64"),
            filename: `preview_${originalFileName}`,
            contentType: outputMimeType,
          },
        },
        extractedContent: extractedContent,
        extractionStatus: "success",
      });
    }

    // Standard response for normal anonymization without preview
    const filename = `phi_anonymized_${originalFileName}`;

    res.setHeader("Content-Type", outputMimeType);
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
    res.setHeader("Content-Length", outputBuffer.length);

    res.send(outputBuffer);
  } catch (error) {
    console.error("Error processing file:", error);

    if (error.message.includes("Only")) {
      return res.status(400).json({
        error:
          "Invalid file type. Please upload a supported spreadsheet file (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb).",
      });
    }

    res.status(500).json({
      error: "Internal server error occurred while processing the file.",
    });
  }
};

const extractFileContent = async (workbook, originalFileName, datasetTitle = "") => {
    /**
     * Extract content from anonymized file for enhanced search
     * Returns extracted content as text
     */
    try {
        const ContentExtractor = require('./contentExtractorController');
        const extractedContent = ContentExtractor.extractSpreadsheetContent(
            workbook,
            datasetTitle
        );
        return extractedContent;
    } catch (error) {
        console.error('Error extracting content:', error);
        return ""; // Return empty string on error, don't fail the upload
    }
};

module.exports = {
  anonymizeFile,
  upload,
};
