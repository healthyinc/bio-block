const multer = require("multer");
const { Worker } = require("worker_threads");
const path = require("path");

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

    // Offload heavy processing to a Worker Thread
    const workerPath = path.join(__dirname, "../workers/anonymizeWorker.js");

    const worker = new Worker(workerPath);

    const transferList = [req.file.buffer.buffer];

    worker.postMessage(
      {
        fileBuffer: req.file.buffer,
        walletAddress,
        generatePreview,
        originalFileName: req.file.originalname,
      },
      transferList
    );

    worker.on("message", (result) => {
      if (!result.success) {
        console.error("Worker error:", result.error);
        return res.status(400).json({
          error: result.error || "Failed to process spreadsheet file.",
        });
      }

      const { outputBuffer, previewBuffer, outputMimeType, originalFileName } = result;

      if (generatePreview && previewBuffer) {
        // Return both files as JSON response
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

      // Standard response for normal anonymization without preview
      const filename = `phi_anonymized_${originalFileName}`;
      const buffer = Buffer.from(outputBuffer);

      res.setHeader("Content-Type", outputMimeType);
      res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
      res.setHeader("Content-Length", buffer.length);

      res.send(buffer);
    });

    worker.on("error", (err) => {
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

module.exports = {
  anonymizeFile,
  upload,
};
