const multer = require('multer');
const XLSX = require('xlsx');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        // Accept Excel files, CSV, ODS, TSV and other spreadsheet formats
        const allowedMimeTypes = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
            'application/vnd.ms-excel', // .xls
            'text/csv', // .csv
            'application/csv', // .csv (alternative)
            'application/vnd.oasis.opendocument.spreadsheet', // .ods
            'text/tab-separated-values', // .tsv
            'application/vnd.ms-excel.sheet.macroEnabled.12', // .xlsm
            'application/vnd.ms-excel.sheet.binary.macroEnabled.12' // .xlsb
        ];
        
        // Also check file extension as a fallback
        const allowedExtensions = /\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i;
        
        if (allowedMimeTypes.includes(file.mimetype) || allowedExtensions.test(file.originalname)) {
            cb(null, true);
        } else {
            cb(new Error('Only spreadsheet files (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb) are allowed'), false);
        }
    },
    limits: {
        fileSize: 10 * 1024 * 1024 * 1024 // 10GB limit
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

const anonymizeFile = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file uploaded. Please upload a spreadsheet file.' 
            });
        }

        const walletAddress = req.body.walletAddress;
        const generatePreview = req.body.generatePreview === 'true';
        const isPersonalData = !!walletAddress;

        console.log('Processing anonymization:', { 
            isPersonalData, 
            generatePreview,
            fileName: req.file.originalname,
            fileSize: req.file.size,
            mimeType: req.file.mimetype,
            walletAddress: walletAddress ? `${walletAddress.substring(0, 6)}...` : 'N/A' 
        });

        // Parse the file using XLSX library which supports multiple formats
        let workbook;
        try {
            // XLSX library automatically detects format based on file content
            workbook = XLSX.read(req.file.buffer, { 
                type: 'buffer',
                cellDates: true,
                cellNF: false,
                cellText: false
            });
        } catch (parseError) {
            console.error('File parsing error:', parseError);
            return res.status(400).json({ 
                error: 'Failed to parse spreadsheet file. Please ensure the file is not corrupted and is in a supported format.' 
            });
        }
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
        
        // Handle all 4 cases:
        // 1. Personal + Patient ID exists: Hash each patient ID individually
        // 2. Personal + No Patient ID: Use wallet address for all rows
        // 3. Institution + Patient ID exists: Hash each patient ID individually  
        // 4. Institution + No Patient ID: Use UUID for each row
        
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
                    // Handle sheets without patient ID column
                    // Case 2: Personal + No Patient ID - Use wallet address
                    // Case 4: Institution + No Patient ID - Use UUID for each row
                    for (let i = 1; i < cleanedData.length; i++) {
                        const row = cleanedData[i];
                        if (row && row.some(cell => cell !== undefined && cell !== null && cell !== '')) {
                            let id;
                            if (isPersonalData && walletAddress) {
                                // Case 2: Personal data without Patient ID - use wallet address
                                id = generateAnonymizedId(walletAddress, 1);
                            } else {
                                // Case 4: Institution data without Patient ID - use UUID
                                const uuid = generateUUID();
                                id = generateAnonymizedId(uuid, i);
                            }
                            
                            columnsToMask.forEach(colIndex => {
                                if (row[colIndex] !== undefined) {
                                    row[colIndex] = id;
                                }
                            });
                        }
                    }
                } else {
                    // Handle sheets with patient ID column
                    // Case 1: Personal + Patient ID exists - Hash each patient ID individually
                    // Case 3: Institution + Patient ID exists - Hash each patient ID individually
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

        // Determine output format based on input file extension
        const originalFileName = req.file.originalname;
        const fileExtension = originalFileName.split('.').pop().toLowerCase();
        
        let outputFormat = 'xlsx'; // Default format
        let outputMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
        
        // Map file extensions to XLSX library book types
        switch (fileExtension) {
            case 'xls':
                outputFormat = 'xls';
                outputMimeType = 'application/vnd.ms-excel';
                break;
            case 'csv':
                outputFormat = 'csv';
                outputMimeType = 'text/csv';
                break;
            case 'ods':
                outputFormat = 'ods';
                outputMimeType = 'application/vnd.oasis.opendocument.spreadsheet';
                break;
            case 'tsv':
                outputFormat = 'txt'; // XLSX library uses 'txt' for TSV
                outputMimeType = 'text/tab-separated-values';
                break;
            case 'xlsm':
                outputFormat = 'xlsm';
                outputMimeType = 'application/vnd.ms-excel.sheet.macroEnabled.12';
                break;
            case 'xlsb':
                outputFormat = 'xlsb';
                outputMimeType = 'application/vnd.ms-excel.sheet.binary.macroEnabled.12';
                break;
            default:
                outputFormat = 'xlsx';
                outputMimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
        }

        const outputBuffer = XLSX.write(cleanedWorkbook, { 
            bookType: outputFormat, 
            type: 'buffer' 
        });

        // Generate preview if requested
        if (generatePreview) {
            const previewWorkbook = XLSX.utils.book_new();
            
            cleanedWorkbook.SheetNames.forEach(sheetName => {
                const worksheet = cleanedWorkbook.Sheets[sheetName];
                const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
                
                if (jsonData.length === 0) {
                    XLSX.utils.book_append_sheet(previewWorkbook, worksheet, sheetName);
                    return;
                }
                
                // Calculate 5% of rows (minimum 5 rows, maximum 50 rows)
                const totalRows = jsonData.length;
                const previewRows = Math.max(5, Math.min(50, Math.ceil(totalRows * 0.05)));
                
                // Extract first 5% of anonymized data including headers
                const previewData = jsonData.slice(0, previewRows);
                
                const previewWorksheet = XLSX.utils.aoa_to_sheet(previewData);
                XLSX.utils.book_append_sheet(previewWorkbook, previewWorksheet, sheetName);
            });

            const previewBuffer = XLSX.write(previewWorkbook, { 
                bookType: outputFormat, 
                type: 'buffer' 
            });

            const extractedContent = await extractFileContent(
                cleanedWorkbook,
                req.file.originalname,
                req.body.datasetTitle || ''
            );

            // Return both files as JSON response
            const timestamp = Date.now();
            return res.json({
                success: true,
                message: 'File anonymized successfully with preview',
                files: {
                    main: {
                        data: outputBuffer.toString('base64'),
                        filename: `anonymized_${originalFileName}`,
                        contentType: outputMimeType
                    },
                    preview: {
                        data: previewBuffer.toString('base64'),
                        filename: `preview_${originalFileName}`,
                        contentType: outputMimeType
                    }
                },
                extractedContent: extractedContent,  // Add this line
                extractionStatus: "success"
            });
        }

        // Standard response for normal anonymization without preview
        const filename = `phi_anonymized_${originalFileName}`;
        
        res.setHeader('Content-Type', outputMimeType);
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.setHeader('Content-Length', outputBuffer.length);

        res.send(outputBuffer);

    } catch (error) {
        console.error('Error processing file:', error);
        
        if (error.message.includes('Only')) {
            return res.status(400).json({ 
                error: 'Invalid file type. Please upload a supported spreadsheet file (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb).' 
            });
        }
        
        res.status(500).json({ 
            error: 'Internal server error occurred while processing the file.' 
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
    upload
};
