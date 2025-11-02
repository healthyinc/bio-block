const axios = require('axios');
const FormData = require('form-data');

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

    // 3. Make the POST request to the Python /anonymize_image endpoint
    const response = await axios.post(`${PYTHON_BACKEND_URL}//simple_preview`, form, {
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