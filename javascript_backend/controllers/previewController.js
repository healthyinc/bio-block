const axios = require('axios');
const FormData = require('form-data');
const XLSX = require('xlsx');

// Get the Python backend URL from environment variables, with a fallback
const PYTHON_BACKEND_URL = process.env.REACT_APP_PYTHON_BACKEND_URL || 'http://localhost:3002';

exports.getImagePreview = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: 'No file uploaded.' });
  }

  try {
    // 1. Create a new FormData object to send to the Python backend
    const form = new FormData();
    
    // 2. Append the file buffer from multer
    // We must pass the original filename and content type
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // 3. Make the POST request to the Python /simple_preview endpoint
    const response = await axios.post(`${PYTHON_BACKEND_URL}/simple_preview`, form, {
      headers: {
        ...form.getHeaders(), // This sets the 'Content-Type: multipart/form-data' correctly
      },
      responseType: 'arraybuffer', // Get the image back as raw data (a buffer)
    });

    // 4. Send the anonymized image data back to the React frontend
    // We must set the content-type header to match the image
    res.set('Content-Type', response.headers['content-type']);
    res.send(response.data);

  } catch (error) {
    console.error('Error in image preview proxy:', error.message);
    res.status(500).json({ message: 'Error generating image preview.' });
  }
};

exports.getSpreadsheetPreview = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: 'No file uploaded.' });
  }

  try {
    // Parse the spreadsheet file
    const workbook = XLSX.read(req.file.buffer, { 
      type: 'buffer',
      cellDates: true,
      cellNF: false,
      cellText: false
    });

    // Get the first sheet
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];
    
    // Convert to JSON with headers in first row
    const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    
    if (jsonData.length === 0) {
      return res.json({
        fileName: req.file.originalname,
        headers: [],
        data: [],
        totalRows: 0,
        previewRows: 0,
        sheetName: firstSheetName
      });
    }

    // Get headers from first row
    const headers = jsonData[0] || [];
    const dataRows = jsonData.slice(1) || [];
    
    // Calculate 5% of rows (minimum 5 rows, maximum 50 rows)
    const totalRows = dataRows.length;
    const previewRows = Math.max(5, Math.min(50, Math.ceil(totalRows * 0.05)));
    
    // Extract first 5% of data for preview
    const previewData = dataRows.slice(0, previewRows);

    return res.json({
      fileName: req.file.originalname,
      headers: headers,
      data: previewData,
      totalRows: totalRows,
      previewRows: previewRows,
      sheetName: firstSheetName,
      message: `Preview showing first ${previewRows} of ${totalRows} rows (${((previewRows/totalRows)*100).toFixed(1)}%)`
    });

  } catch (error) {
    console.error('Error generating spreadsheet preview:', error.message);
    res.status(500).json({ message: 'Error generating spreadsheet preview.' });
  }
};

exports.getPdfPreview = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: 'No file uploaded.' });
  }

  try {
    // For PDF preview, we simply return the PDF file as-is for browser display
    // The browser will handle rendering it in an iframe or embed tag
    res.set('Content-Type', 'application/pdf');
    res.set('Content-Disposition', `inline; filename="${req.file.originalname}"`);
    res.send(req.file.buffer);
  } catch (error) {
    console.error('Error generating PDF preview:', error.message);
    res.status(500).json({ message: 'Error generating PDF preview.' });
  }
};

exports.getDicomPreview = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: 'No file uploaded.' });
  }

  try {
    // 1. Create a new FormData object to send to the Python backend
    const form = new FormData();
    
    // 2. Append the file buffer from multer
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // 3. Make the POST request to the Python /preview_dicom endpoint
    const response = await axios.post(`${PYTHON_BACKEND_URL}/preview_dicom`, form, {
      headers: {
        ...form.getHeaders(),
      },
      responseType: 'arraybuffer',
    });

    // 4. Send the converted image data back to the React frontend
    res.set('Content-Type', response.headers['content-type'] || 'image/png');
    res.send(response.data);

  } catch (error) {
    console.error('Error in DICOM preview proxy:', error.message);
    res.status(500).json({ message: 'Error generating DICOM preview.' });
  }
};