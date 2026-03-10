const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const os = require("os");
const crypto = require("crypto");

// Configure multer for disk storage with file-type validation
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
    // Accept health data formats and encrypted blobs that Bio-Block handles
    const allowedMimeTypes = [
      // Encrypted blobs (primary use case for IPFS upload)
      "application/octet-stream",
      // Spreadsheet formats
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
      "application/vnd.ms-excel", // .xls
      "text/csv", // .csv
      "application/csv", // .csv (alternative)
      "application/vnd.oasis.opendocument.spreadsheet", // .ods
      "text/tab-separated-values", // .tsv
      // Image formats
      "image/jpeg", // .jpg/.jpeg
      "image/png", // .png
      // Document formats
      "application/pdf", // .pdf
      // Medical imaging
      "application/dicom", // .dcm
    ];

    // Extension fallback for cases where MIME type is unreliable
    const allowedExtensions =
      /\.(enc|xlsx|xls|csv|ods|tsv|jpg|jpeg|png|pdf|dcm|nii|nii\.gz)$/i;

    if (
      allowedMimeTypes.includes(file.mimetype) ||
      allowedExtensions.test(file.originalname)
    ) {
      cb(null, true);
    } else {
      cb(
        new Error(
          "File type not allowed. Accepted: encrypted blobs, spreadsheets (.xlsx, .csv, .ods, .tsv), images (.jpg, .png), documents (.pdf), and medical imaging (.dcm, .nii)."
        ),
        false
      );
    }
  },
  limits: {
    fileSize: 2 * 1024 * 1024 * 1024, // 2GB
  },
});

const uploadToIPFS = async (req, res) => {
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
        error: "No file uploaded.",
      });
    }

    const { fileName } = req.body;

    console.log("Uploading to IPFS:", {
      fileName,
      fileSize: req.file.size,
    });

    const fileStream = fs.createReadStream(req.file.path);

    const formData = new FormData();
    formData.append("file", fileStream, {
      filename: fileName || "encrypted_file",
      contentType: "application/octet-stream",
    });

    const pinataMetadata = JSON.stringify({
      name: fileName || "Encrypted Document",
      keyvalues: {
        encrypted: "true",
        uploadedAt: new Date().toISOString(),
      },
    });
    formData.append("pinataMetadata", pinataMetadata);

    const pinataOptions = JSON.stringify({
      cidVersion: 0,
    });
    formData.append("pinataOptions", pinataOptions);

    const pinataResponse = await axios.post(
      "https://api.pinata.cloud/pinning/pinFileToIPFS",
      formData,
      {
        maxBodyLength: "Infinity",
        headers: {
          "Content-Type": `multipart/form-data; boundary=${formData._boundary}`,
          Authorization: `Bearer ${process.env.PINATA_JWT}`,
          ...formData.getHeaders(),
        },
      }
    );

    const ipfsHash = pinataResponse.data.IpfsHash;

    console.log("IPFS upload successful:", { ipfsHash, fileName });

    cleanupTempFile();

    res.json({
      success: true,
      ipfsHash: ipfsHash,
      fileName: fileName,
      fileSize: req.file.size,
    });
  } catch (error) {
    cleanupTempFile();
    console.error("IPFS upload error:", error.response?.data || error.message);

    if (error.response?.status === 401) {
      return res.status(401).json({
        error: "Invalid Pinata API credentials",
      });
    }

    res.status(500).json({
      error: "IPFS upload failed: " + (error.response?.data?.error || error.message),
    });
  }
};

module.exports = {
  uploadToIPFS,
  upload,
};
