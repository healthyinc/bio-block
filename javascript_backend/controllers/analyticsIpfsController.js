const axios = require("axios");

// Upload a JSON analytics result to IPFS via Pinata.
const uploadAnalyticsResult = async (req, res) => {
  try {
    const { resultJson, fileName } = req.body;

    if (!resultJson || typeof resultJson !== "object") {
      return res.status(400).json({
        error: "Missing or invalid 'resultJson' in request body.",
      });
    }

    const jsonString = JSON.stringify(resultJson, null, 2);
    const buffer = Buffer.from(jsonString, "utf-8");

    const FormData = require("form-data");
    const formData = new FormData();
    formData.append("file", buffer, {
      filename: fileName || "analytics-result.json",
      contentType: "application/json",
    });

    const pinataMetadata = JSON.stringify({
      name: fileName || "Analytics Result",
      keyvalues: {
        type: "analytics-result",
        analysisType: resultJson.analysis_type || "unknown",
        sourceCID: resultJson.source_dataset_cid || "",
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

    console.log("Analytics result uploaded to IPFS:", {
      ipfsHash,
      fileName,
      size: buffer.length,
    });

    res.json({
      success: true,
      ipfsHash: ipfsHash,
      fileName: fileName || "analytics-result.json",
      fileSize: buffer.length,
    });
  } catch (error) {
    console.error("Analytics IPFS upload error:", error.response?.data || error.message);

    if (error.response?.status === 401) {
      return res.status(401).json({
        error: "Invalid Pinata API credentials",
      });
    }

    res.status(500).json({
      error: "Analytics IPFS upload failed: " + (error.response?.data?.error || error.message),
    });
  }
};

module.exports = { uploadAnalyticsResult };
