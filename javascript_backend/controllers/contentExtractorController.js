const XLSX = require('xlsx');

/**
 * Content Extractor for Enhanced Retrieval
 * Extracts meaningful content from files for semantic search indexing
 */

class ContentExtractor {
    /**
     * Extract content from spreadsheet workbook
     * @param {Object} workbook - XLSX workbook object
     * @param {string} datasetTitle - Title of the dataset
     * @returns {string} Extracted content for vectorization
     */
    static extractSpreadsheetContent(workbook, datasetTitle = "") {
        const contentParts = [];
        
        if (datasetTitle) {
            contentParts.push(`Dataset: ${datasetTitle}`);
        }
        
        workbook.SheetNames.forEach((sheetName, sheetIdx) => {
            contentParts.push(`\nSheet: ${sheetName}`);
            
            const worksheet = workbook.Sheets[sheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
            
            if (jsonData.length === 0) return;
            
            const headers = jsonData[0] || [];
            contentParts.push(`Columns: ${headers.join(', ')}`);
            
            // Extract sample rows (first 20 rows) for content understanding
            const sampleSize = Math.min(20, jsonData.length - 1);
            for (let i = 1; i <= sampleSize; i++) {
                const row = jsonData[i];
                const rowContent = [];
                
                for (let colIdx = 0; colIdx < headers.length && colIdx < row.length; colIdx++) {
                    const cell = row[colIdx];
                    const header = headers[colIdx];
                    
                    // Skip anonymized IDs and empty cells
                    if (cell && header && !String(cell).startsWith('WID_')) {
                        rowContent.push(`${header}: ${cell}`);
                    }
                }
                
                if (rowContent.length > 0) {
                    contentParts.push(`Row ${i}: ${rowContent.join('; ')}`);
                }
            }
        });
        
        return contentParts.join('\n');
    }
    
    /**
     * Extract content from CSV text
     * @param {string} csvText - CSV content
     * @param {string} datasetTitle - Title of the dataset
     * @returns {string} Extracted content
     */
    static extractCSVContent(csvText, datasetTitle = "") {
        const contentParts = [];
        
        if (datasetTitle) {
            contentParts.push(`Dataset: ${datasetTitle}`);
        }
        
        const lines = csvText.trim().split('\n');
        if (lines.length === 0) return "";
        
        const headers = lines[0].split(',').map(h => h.trim());
        contentParts.push(`Columns: ${headers.join(', ')}`);
        
        // Sample first 20 rows
        const sampleSize = Math.min(20, lines.length - 1);
        for (let i = 1; i <= sampleSize; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const rowContent = [];
            
            for (let colIdx = 0; colIdx < headers.length && colIdx < values.length; colIdx++) {
                const value = values[colIdx];
                const header = headers[colIdx];
                
                if (value && header && !value.startsWith('WID_')) {
                    rowContent.push(`${header}: ${value}`);
                }
            }
            
            if (rowContent.length > 0) {
                contentParts.push(`Row ${i}: ${rowContent.join('; ')}`);
            }
        }
        
        return contentParts.join('\n');
    }
}

module.exports = ContentExtractor;