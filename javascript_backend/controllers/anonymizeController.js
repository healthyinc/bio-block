const multer = require("multer");
const { Worker } = require("worker_threads");
const path = require("path");
const fs = require("fs");
const os = require("os");
const crypto = require("crypto");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, os.tmpdir());
  },
  filename: (req, file, cb) => {
    const uniqueName = `upload-${crypto.randomUUID()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
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

    const allowedExtensions = /\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i;

    if (allowedMimeTypes.includes(file.mimetype) || allowedExtensions.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error("Only spreadsheet files (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb) are allowed"), false);
    }
  },
  limits: {
    fileSize: 20 * 1024 * 1024, // 20MB
  },
});

const anonymizeFile = async (req, res) => {
  const cleanupTempFile = () => {
    if (req.file && req.file.path) {
      fs.unlink(req.file.path, (err) => {
        if (err && err.code !== "ENOENT") {
          console.error("Failed to clean up temp file:", err);
        }
      });
    }
  };

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

    const diskBuffer = fs.readFileSync(req.file.path);

    const arrayBuffer = new ArrayBuffer(diskBuffer.length);
    const fileBuffer = Buffer.from(arrayBuffer);
    diskBuffer.copy(fileBuffer);

    const workerPath = path.join(__dirname, "../workers/anonymizeWorker.js");

    const worker = new Worker(workerPath);

    const transferList = [arrayBuffer];

    worker.postMessage(
      {
        fileBuffer: fileBuffer,
        walletAddress,
        generatePreview,
        originalFileName: req.file.originalname,
      },
      transferList
    );

    worker.on("message", (result) => {
      cleanupTempFile();

      if (!result.success) {
        console.error("Worker error:", result.error);
        return res.status(400).json({
          error: result.error || "Failed to process spreadsheet file.",
        });
      }

      const { outputBuffer, previewBuffer, outputMimeType, originalFileName } = result;

      if (generatePreview && previewBuffer) {
        return res.json({
          success: true,
          message: "File anonymized successfully with preview",
          files: {
            main: {
              data: Buffer.from(outputBuffer).toString("base64"),
              filename: `anonymized_${originalFileName}`,
              contentType: outputMimeType,
            },
            preview: {
              data: Buffer.from(previewBuffer).toString("base64"),
              filename: `preview_${originalFileName}`,
              contentType: outputMimeType,
            },
          },
        });
      }

      const filename = `phi_anonymized_${originalFileName}`;
      const buffer = Buffer.from(outputBuffer);

      res.setHeader("Content-Type", outputMimeType);
      res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
      res.setHeader("Content-Length", buffer.length);

      res.send(buffer);
    });

    worker.on("error", (err) => {
      cleanupTempFile();
      console.error("Worker thread error:", err);
      res.status(500).json({
        error: "Internal server error occurred while processing the file.",
      });
    });

    worker.on("exit", (code) => {
      if (code !== 0) {
        console.error(new Error(`Worker stopped with exit code ${code}`));
      }
    });
  } catch (error) {
    cleanupTempFile();
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
    const ContentExtractor = require("./contentExtractorController");
    const extractedContent = ContentExtractor.extractSpreadsheetContent(workbook, datasetTitle);
    return extractedContent;
  } catch (error) {
    console.error("Error extracting content:", error);
    return ""; // Return empty string on error, don't fail the upload
  }
};

module.exports = {
  anonymizeFile,
  upload,
};
