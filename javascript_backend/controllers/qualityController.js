const multer = require('multer');
const XLSX = require('xlsx');

const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        const allowedMimeTypes = [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'text/csv',
            'application/csv',
            'application/vnd.oasis.opendocument.spreadsheet',
            'text/tab-separated-values',
            'application/vnd.ms-excel.sheet.macroEnabled.12',
            'application/vnd.ms-excel.sheet.binary.macroEnabled.12'
        ];
        
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

function inferType(value) {
    if (value === null || value === undefined || value === '') return 'empty';
    if (!isNaN(Date.parse(value)) && typeof value === 'string' && value.includes('-')) return 'date';
    if (!isNaN(Number(value))) return 'number';
    if (typeof value === 'boolean' || value === 'true' || value === 'false') return 'boolean';
    return 'text';
}

const analyzeDatasetQuality = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No file uploaded. Please upload a spreadsheet file.' 
            });
        }

        let workbook;
        try {
            workbook = XLSX.read(req.file.buffer, { 
                type: 'buffer',
                cellDates: true,
                cellNF: false,
                cellText: false
            });
        } catch (parseError) {
            console.error('File parsing error:', parseError);
            return res.status(400).json({ 
                error: 'Failed to parse spreadsheet file. Please ensure it is a supported format.' 
            });
        }

        const report = {
            fileName: req.file.originalname,
            fileSize: req.file.size,
            sheets: []
        };

        workbook.SheetNames.forEach(sheetName => {
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
                name: header || 'Unknown',
                missingCount: 0,
                types: { number: 0, text: 0, boolean: 0, date: 0 }
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
                    
                    if (type === 'empty') {
                        columnStats[j].missingCount++;
                    } else {
                        columnStats[j].types[type] = (columnStats[j].types[type] || 0) + 1;
                    }
                }
            }

            // Summarize columns
            const columns = columnStats.map(stat => {
                // Find dominant type
                let dominantType = 'text';
                let maxCount = 0;
                for (const [type, count] of Object.entries(stat.types)) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominantType = type;
                    }
                }

                const missingPercentage = totalRows > 0 ? ((stat.missingCount / totalRows) * 100).toFixed(2) : 0;

                return {
                    name: stat.name,
                    inferredType: maxCount === 0 ? 'empty' : dominantType,
                    missingCount: stat.missingCount,
                    missingPercentage: parseFloat(missingPercentage)
                };
            });

            report.sheets.push({
                name: sheetName,
                totalRows,
                totalColumns,
                duplicateRows: duplicateCount,
                duplicatePercentage: totalRows > 0 ? parseFloat(((duplicateCount / totalRows) * 100).toFixed(2)) : 0,
                columns
            });
        });

        res.json({
            success: true,
            report
        });

    } catch (error) {
        console.error('Quality analysis block error:', error);
        res.status(500).json({ error: 'Internal server error analyzing quality.' });
    }
};

module.exports = {
    analyzeDatasetQuality,
    upload
};
