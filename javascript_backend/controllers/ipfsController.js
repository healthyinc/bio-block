const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
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
