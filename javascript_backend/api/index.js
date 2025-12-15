require('dotenv').config();
const express = require("express");
const axios = require("axios");
const { QdrantClient } = require("@qdrant/js-client-rest");
const cors = require("cors");
const multer = require('multer');
const XLSX = require('xlsx');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

const app = express();

app.use(cors());
app.use(express.json());

// const qdrant = new QdrantClient({
//      url: process.env.QDRANT_URL,
//     apiKey: process.env.QDRANT_API_KEY,
// });
// const COLLECTION_NAME = "new_user_data";

// async function ensureCollection() {
//     try {
//         await qdrant.getCollection(COLLECTION_NAME);
//     } catch {
//         await qdrant.createCollection(COLLECTION_NAME, {
//             vectors: {
//                 size: 384, 
//                 distance: "Cosine"
//             }
//         });
//     }
// }
// ensureCollection();

// const generateId = () => {
//   const timestamp = Date.now();
//   const randomSuffix = Math.floor(Math.random() * 10000);
//   return Number(`${timestamp}${randomSuffix}`);
// };

const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || 
            file.mimetype === 'application/vnd.ms-excel') {
            cb(null, true);
        } else {
            cb(new Error('Only Excel files (.xlsx, .xls) are allowed'), false);
        }
    },
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

const phiKeywords = [
    'dob', 'date of birth', 'address', 'phone', 'mobile', 'email',
    'ssn', 'social security', 'mrn', 'medical record', 'health plan',
    'license', 'account number', 'ip address', 'device id', 'biometric',
    'photo', 'facial', 'fingerprint', 'signature', 'first name', 'last name',
    'name'
];

function generateAnonymizedId(input, index) {
    const hash = crypto.createHash('sha256').update(input).digest('hex');
    const shortHash = hash.substring(0, 8);
    return `WID_${shortHash}`;
}

function generateUUID() {
    return uuidv4();
}

// app.post("/store", async (req, res) => {
//     const { summary, cid, metadata } = req.body;

//     if (!summary || !cid) {
//         return res.status(400).json({ error: "Missing 'summary' or 'cid'" });
//     }

//     try {
//         const { data } = await axios.post("http://localhost:8000/embed", { text: summary });
//         const embedding = data.embedding;
        
//         if (!embedding || embedding.length !== 384) {
//             return res.status(500).json({ error: "Invalid embedding received" });
//         }
        
//         await qdrant.upsert(COLLECTION_NAME, {
//             points: [
//                 {
//                     id: generateId(),
//                     vector: embedding,
//                     payload: { summary, cid, ...metadata }
//                 }
//             ]
//         });

//         res.status(200).json({ message: "Stored successfully", cid });
//     } catch (err) {
//         console.error("Store Error:", err.message);
//         res.status(500).json({ error: "Failed to store data" });
//     }
// });

// app.post("/search", async (req, res) => {
//     const { query } = req.body;
//     if (!query) {
//         return res.status(400).json({ error: "Missing 'query'" });
//     }

//     try {
//         const { data } = await axios.post("http://localhost:8000/embed", { text: query });
//         const queryVector = data.embedding;

//         const searchResult = await qdrant.search(COLLECTION_NAME, {
//             vector: queryVector,
//             limit: 5,
//             with_payload: true
//         });

//         const results = searchResult.map((item) => ({
//             cid: item.payload.cid,
//             score: item.score,
//             metadata: item.payload
//         }));

//         res.status(200).json({ results });
//     } catch (err) {
//         console.error("Search Error:", err.message);
//         res.status(500).json({ error: "Failed to search" });
//     }
// });

app.post('/anonymize', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file uploaded. Please upload an Excel file.' 
            });
        }

    
        const workbook = XLSX.read(req.file.buffer, { type: 'buffer' });

        const allIdentifiers = new Set();
        const sheetColumnRefs = {};

     
        workbook.SheetNames.forEach(sheetName => {
            const worksheet = workbook.Sheets[sheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
            
            if (jsonData.length === 0) return;
            
            const headers = jsonData[0] || [];
            let patientIdCol = null;

         
            headers.forEach((header, index) => {
                if (header && typeof header === 'string') {
                    const headerLower = header.toLowerCase();
                    if (headerLower.includes('patient') && headerLower.includes('id')) {
                        patientIdCol = index;
                    }
                }
            });

            if (patientIdCol !== null) {
           
                for (let i = 1; i < jsonData.length; i++) {
                    const row = jsonData[i];
                    if (row && row[patientIdCol]) {
                        const patientId = String(row[patientIdCol]).toLowerCase().trim();
                        allIdentifiers.add(patientId);
                    }
                }
                sheetColumnRefs[sheetName] = { patientIdCol };
            } else {
              
                for (let i = 1; i < jsonData.length; i++) {
                    const row = jsonData[i];
                    if (row && row.some(cell => cell !== undefined && cell !== null && cell !== '')) {
                        const uuid = generateUUID();
                        allIdentifiers.add(uuid);
                    }
                }
                sheetColumnRefs[sheetName] = { useUUID: true };
            }
        });

      
        const sortedIdentifiers = Array.from(allIdentifiers).sort();
        const identifierToId = {};
        sortedIdentifiers.forEach((identifier, index) => {
            identifierToId[identifier] = generateAnonymizedId(identifier, index + 1);
        });


        const cleanedWorkbook = XLSX.utils.book_new();

        workbook.SheetNames.forEach(sheetName => {
            const worksheet = workbook.Sheets[sheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
            
            if (jsonData.length === 0) {
                XLSX.utils.book_append_sheet(cleanedWorkbook, worksheet, sheetName);
                return;
            }

            const headers = jsonData[0] || [];
            const cleanedData = jsonData.map(row => [...row]);

            if (sheetColumnRefs[sheetName]) {
                const sheetRefs = sheetColumnRefs[sheetName];

            
                const columnsToMask = [];
                headers.forEach((header, index) => {
                    if (header && typeof header === 'string') {
                        const headerLower = header.toLowerCase();
                        const isPatientIdColumn = index === sheetRefs.patientIdCol;
                        const isPhiColumn = phiKeywords.some(keyword => {
                            return headerLower.includes(keyword) || 
                                   headerLower.replace(/\s+/g, '').includes(keyword.replace(/\s+/g, ''));
                        });
                        
                        if (isPatientIdColumn || isPhiColumn) {
                            columnsToMask.push(index);
                        }
                    }
                });

                if (sheetRefs.useUUID) {
                  
                    for (let i = 1; i < cleanedData.length; i++) {
                        const row = cleanedData[i];
                        if (row && row.some(cell => cell !== undefined && cell !== null && cell !== '')) {
                            const uuid = generateUUID();
                            const id = generateAnonymizedId(uuid, i);
                            
                            columnsToMask.forEach(colIndex => {
                                if (row[colIndex] !== undefined) {
                                    row[colIndex] = id;
                                }
                            });
                        }
                    }
                } else {
                
                    for (let i = 1; i < cleanedData.length; i++) {
                        const row = cleanedData[i];
                        let id = null;

                        if (sheetRefs.patientIdCol !== undefined && row && row[sheetRefs.patientIdCol]) {
                            const patientId = String(row[sheetRefs.patientIdCol]).toLowerCase().trim();
                            id = identifierToId[patientId];
                        }

                        if (id) {
                            columnsToMask.forEach(colIndex => {
                                if (row[colIndex] !== undefined) {
                                    row[colIndex] = id;
                                }
                            });
                        }
                    }
                }
            }

            const newWorksheet = XLSX.utils.aoa_to_sheet(cleanedData);
            XLSX.utils.book_append_sheet(cleanedWorkbook, newWorksheet, sheetName);
        });

     
        const outputBuffer = XLSX.write(cleanedWorkbook, { 
            bookType: 'xlsx', 
            type: 'buffer' 
        });

   
        const timestamp = Date.now();
        const filename = `phi_anonymized_${timestamp}.xlsx`;
        
        res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.setHeader('Content-Length', outputBuffer.length);

       
        res.send(outputBuffer);

    } catch (error) {
        if (error.message && error.message.includes('Only Excel files')) {
            return res.status(400).json({ 
                error: 'Invalid file type. Please upload an Excel file (.xlsx or .xls).' 
            });
        }
        
        res.status(500).json({ 
            error: 'Internal server error occurred while processing the file.' 
        });
    }
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        message: 'Backend API is running',
        timestamp: new Date().toISOString()
    });
});

app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ 
                error: 'File too large. Maximum size is 10MB.' 
            });
        }
    }
    
    res.status(500).json({ 
        error: 'Internal server error' 
    });
});

// Export the Express API for Vercel
module.exports = app;
