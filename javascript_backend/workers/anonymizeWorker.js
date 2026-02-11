const { parentPort } = require('worker_threads');
const XLSX = require('xlsx');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

const phiKeywords = [
    'dob', 'date of birth', 'address', 'phone', 'mobile', 'email',
    'ssn', 'social security', 'mrn', 'medical record', 'health plan',
    'license', 'account number', 'ip address', 'device id', 'biometric',
    'photo', 'facial', 'fingerprint', 'signature', 'first name', 'last name',
    'name'
];

const generateAnonymizedId = (input, index) => {
    const hash = crypto.createHash('sha256').update(input).digest('hex');
    return `WID_${hash.substring(0, 8)}`;
};

parentPort.on('message', (workerData) => {
    try {
        const { fileBuffer, walletAddress, generatePreview, originalFileName } = workerData;
        const isPersonalData = !!walletAddress;

        const workbook = XLSX.read(fileBuffer, {
            type: 'buffer',
            cellDates: true,
            cellNF: false,
            cellText: false
        });

        const allIdentifiers = new Set();
        const sheetColumnRefs = {};

        // First pass: scan for identifiers
        workbook.SheetNames.forEach(name => {
            const sheet = workbook.Sheets[name];
            const data = XLSX.utils.sheet_to_json(sheet, { header: 1 });
            if (!data.length) return;

            const headers = data[0] || [];
            let pidCol = null;

            headers.forEach((h, i) => {
                if (typeof h === 'string' && h.toLowerCase().includes('patient') && h.toLowerCase().includes('id')) {
                    pidCol = i;
                }
            });

            if (pidCol !== null) {
                for (let i = 1; i < data.length; i++) {
                    if (data[i] && data[i][pidCol]) {
                        allIdentifiers.add(String(data[i][pidCol]).toLowerCase().trim());
                    }
                }
                sheetColumnRefs[name] = { pidCol };
            } else {
                // No ID column, generate UUIDs for rows with content
                for (let i = 1; i < data.length; i++) {
                    if (data[i] && data[i].some(c => c !== undefined && c !== null && c !== '')) {
                        allIdentifiers.add(uuidv4());
                    }
                }
                sheetColumnRefs[name] = { useUUID: true };
            }
        });

        const sortedIds = Array.from(allIdentifiers).sort();
        const idMap = {};
        sortedIds.forEach((id, i) => idMap[id] = generateAnonymizedId(id, i + 1));

        const outWb = XLSX.utils.book_new();

        // Second pass: anonymize
        workbook.SheetNames.forEach(name => {
            const sheet = workbook.Sheets[name];
            let data = XLSX.utils.sheet_to_json(sheet, { header: 1 });

            if (!data.length) {
                XLSX.utils.book_append_sheet(outWb, sheet, name);
                return;
            }

            const headers = data[0] || [];
            const cleanData = data.map(r => [...r]);

            if (sheetColumnRefs[name]) {
                const { pidCol, useUUID } = sheetColumnRefs[name];
                const maskCols = [];

                headers.forEach((h, i) => {
                    if (typeof h !== 'string') return;
                    const lower = h.toLowerCase();
                    const isPid = i === pidCol;
                    const isPhi = phiKeywords.some(k =>
                        lower.includes(k) || lower.replace(/\s+/g, '').includes(k.replace(/\s+/g, ''))
                    );
                    if (isPid || isPhi) maskCols.push(i);
                });

                if (useUUID) {
                    for (let i = 1; i < cleanData.length; i++) {
                        const row = cleanData[i];
                        if (row && row.some(c => c !== undefined && c !== null && c !== '')) {
                            const id = isPersonalData && walletAddress
                                ? generateAnonymizedId(walletAddress, 1)
                                : generateAnonymizedId(uuidv4(), i);

                            maskCols.forEach(c => { if (row[c] !== undefined) row[c] = id; });
                        }
                    }
                } else {
                    for (let i = 1; i < cleanData.length; i++) {
                        const row = cleanData[i];
                        let id = null;
                        if (pidCol !== undefined && row && row[pidCol]) {
                            id = idMap[String(row[pidCol]).toLowerCase().trim()];
                        }
                        if (id) {
                            maskCols.forEach(c => { if (row[c] !== undefined) row[c] = id; });
                        }
                    }
                }
            }
            XLSX.utils.book_append_sheet(outWb, XLSX.utils.aoa_to_sheet(cleanData), name);
        });

        const ext = originalFileName.split('.').pop().toLowerCase();
        const typeMap = {
            'xls': ['xls', 'application/vnd.ms-excel'],
            'csv': ['csv', 'text/csv'],
            'ods': ['ods', 'application/vnd.oasis.opendocument.spreadsheet'],
            'tsv': ['txt', 'text/tab-separated-values'],
            'xlsm': ['xlsm', 'application/vnd.ms-excel.sheet.macroEnabled.12'],
            'xlsb': ['xlsb', 'application/vnd.ms-excel.sheet.binary.macroEnabled.12']
        };

        const [bookType, contentType] = typeMap[ext] || ['xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
        const writeOpts = { bookType, type: 'buffer' };

        const outputBuffer = XLSX.write(outWb, writeOpts);
        let previewBuffer = null;

        if (generatePreview) {
            const previewWb = XLSX.utils.book_new();
            outWb.SheetNames.forEach(name => {
                const sheet = outWb.Sheets[name];
                const data = XLSX.utils.sheet_to_json(sheet, { header: 1 });
                if (!data.length) {
                    XLSX.utils.book_append_sheet(previewWb, sheet, name);
                    return;
                }
                // 5% rows, min 5, max 50
                const limit = Math.max(5, Math.min(50, Math.ceil(data.length * 0.05)));
                XLSX.utils.book_append_sheet(previewWb, XLSX.utils.aoa_to_sheet(data.slice(0, limit)), name);
            });
            previewBuffer = XLSX.write(previewWb, writeOpts);
        }

        parentPort.postMessage({
            success: true,
            outputBuffer,
            previewBuffer,
            outputMimeType: contentType,
            originalFileName
        });

    } catch (err) {
        parentPort.postMessage({ success: false, error: err.message });
    }
});
