const multer = require("multer");
const XLSX = require("xlsx");

const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedMimeTypes = [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "text/csv",
      "application/csv",
      "application/vnd.oasis.opendocument.spreadsheet",
      "text/tab-separated-values",
      "application/vnd.ms-excel.sheet.macroEnabled.12",
      "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    ];

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

function inferType(value) {
  if (value === null || value === undefined || value === "") return "empty";
  if (!isNaN(Date.parse(value)) && typeof value === "string" && value.includes("-")) return "date";
  if (!isNaN(Number(value))) return "number";
  if (typeof value === "boolean" || value === "true" || value === "false") return "boolean";
  return "text";
}

const analyzeDatasetQuality = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded. Please upload a spreadsheet file.",
      });
    }

    let workbook;
    try {
      workbook = XLSX.read(req.file.buffer, {
        type: "buffer",
        cellDates: true,
        cellNF: false,
        cellText: false,
      });
    } catch (parseError) {
      console.error("File parsing error:", parseError);
      return res.status(400).json({
        error: "Failed to parse spreadsheet file. Please ensure it is a supported format.",
      });
    }

    const report = {
      fileName: req.file.originalname,
      fileSize: req.file.size,
      sheets: [],
    };

    workbook.SheetNames.forEach((sheetName) => {
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      if (jsonData.length === 0) {
        report.sheets.push({ name: sheetName, totalRows: 0, columns: [] });
        return;
      }

      const headers = jsonData[0] || [];
      const dataRows = jsonData.slice(1);
      const totalRows = dataRows.length;
      const totalColumns = headers.length;

      const columnStats = headers.map((header) => ({
        name: header || "Unknown",
        missingCount: 0,
        types: { number: 0, text: 0, boolean: 0, date: 0 },
        numericValues: [], // New: collect numeric values for stats
      }));

      // Track duplicate rows (naive JSON stringify comparison)
      const rowStrings = new Set();
      let duplicateCount = 0;

      for (let i = 0; i < totalRows; i++) {
        const row = dataRows[i];

        // Duplicate check
        const rowStr = JSON.stringify(row);
        if (rowStrings.has(rowStr)) {
          duplicateCount++;
        } else {
          rowStrings.add(rowStr);
        }

        for (let j = 0; j < totalColumns; j++) {
          const value = row ? row[j] : undefined;
          const type = inferType(value);

          if (type === "empty") {
            columnStats[j].missingCount++;
          } else {
            columnStats[j].types[type] = (columnStats[j].types[type] || 0) + 1;
            if (type === "number") {
              columnStats[j].numericValues.push(Number(value));
            }
          }
        }
      }

      // Summarize columns
      const columns = columnStats.map((stat) => {
        // Find dominant type
        let dominantType = "text";
        let maxCount = 0;
        for (const [type, count] of Object.entries(stat.types)) {
          if (count > maxCount) {
            maxCount = count;
            dominantType = type;
          }
        }

        const missingPercentage =
          totalRows > 0 ? ((stat.missingCount / totalRows) * 100).toFixed(2) : 0;

        const result = {
          name: stat.name,
          inferredType: maxCount === 0 ? "empty" : dominantType,
          missingCount: stat.missingCount,
          missingPercentage: parseFloat(missingPercentage),
        };

        // Add statistical summaries for numeric columns
        if (dominantType === "number" && stat.numericValues.length > 0) {
          const vals = stat.numericValues.sort((a, b) => a - b);
          const sum = vals.reduce((a, b) => a + b, 0);
          const mean = sum / vals.length;

          // Median
          const mid = Math.floor(vals.length / 2);
          const median = vals.length % 2 !== 0 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;

          // Min/Max
          const min = vals[0];
          const max = vals[vals.length - 1];

          // Standard Deviation
          const sqDiffs = vals.map((v) => Math.pow(v - mean, 2));
          const avgSqDiff = sqDiffs.reduce((a, b) => a + b, 0) / vals.length;
          const stdDev = Math.sqrt(avgSqDiff);

          // IQR-based Outlier Detection
          const q1Idx = (vals.length - 1) * 0.25;
          const q3Idx = (vals.length - 1) * 0.75;

          const getPercentile = (idx) => {
            const base = Math.floor(idx);
            const rest = idx - base;
            if (vals[base + 1] !== undefined) {
              return vals[base] + rest * (vals[base + 1] - vals[base]);
            }
            return vals[base];
          };

          const q1 = getPercentile(q1Idx);
          const q3 = getPercentile(q3Idx);
          const iqr = q3 - q1;
          const lowerBound = q1 - 1.5 * iqr;
          const upperBound = q3 + 1.5 * iqr;

          const outliers = vals.filter((v) => v < lowerBound || v > upperBound);

          result.statistics = {
            min: parseFloat(min.toFixed(4)),
            max: parseFloat(max.toFixed(4)),
            mean: parseFloat(mean.toFixed(4)),
            median: parseFloat(median.toFixed(4)),
            stdDev: parseFloat(stdDev.toFixed(4)),
            q1: parseFloat(q1.toFixed(4)),
            q3: parseFloat(q3.toFixed(4)),
            outlierCount: outliers.length,
            outliers: outliers.slice(0, 10), // Limit to first 10 examples
            count: vals.length,
          };
        }

        return result;
      });

      report.sheets.push({
        name: sheetName,
        totalRows,
        totalColumns,
        duplicateRows: duplicateCount,
        duplicatePercentage:
          totalRows > 0 ? parseFloat(((duplicateCount / totalRows) * 100).toFixed(2)) : 0,
        columns,
      });
    });

    res.json({
      success: true,
      report,
    });
  } catch (error) {
    console.error("Quality analysis block error:", error);
    res.status(500).json({ error: "Internal server error analyzing quality." });
  }
};

module.exports = {
  analyzeDatasetQuality,
  upload,
};
