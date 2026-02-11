"use client";

import React, { useState } from 'react';
import {
  Search,
  Filter,
  ArrowLeft,
  Download,
  ShoppingCart,
  Eye,
  FileText,
  Calendar,
  User,
  Building,
  X,
} from 'lucide-react';
import { decryptFile } from '../lib/encryptionUtils';
import { purchaseDocument, getDocumentPrice } from '../lib/contractService';
import StreamingEncryption from '../lib/streamingEncryption';

// Types
interface SearchDataProps {
  onBack: () => void;
}

interface FilterState {
  dataType: string;
  gender: string;
  ageRange: string;
  dataSource: string;
  fileType: string;
}

interface DocumentMetadata {
  fileName?: string;
  fileType?: string;
  fileSize?: number;
  uploadDate?: string;
  dataSource?: string;
  gender?: string;
  dataType?: string;
  price?: string;
  previewHash?: string;
}

interface SearchResult {
  cid?: string;
  ipfsHash?: string;
  hash?: string;
  summary?: string;
  description?: string;
  content?: string;
  document?: string;
  title?: string;
  fileName?: string;
  metadata?: DocumentMetadata;
}

interface PreviewData {
  fileName?: string;
  fileSize?: number;
  dataType?: string;
  message?: string;
  headers?: string[];
  data?: unknown[][];
  totalRows?: number;
  totalColumns?: number;
  sheetName?: string;
  error?: string;
}

export default function SearchData({ onBack }: SearchDataProps) {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [downloading, setDownloading] = useState<Record<number, boolean>>({});
  const [purchasing, setPurchasing] = useState<Record<number, boolean>>({});

  // Preview modal states
  const [showPreviewModal, setShowPreviewModal] = useState<boolean>(false);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);

  // Filter states
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [filters, setFilters] = useState<FilterState>({
    dataType: '',
    gender: '',
    ageRange: '',
    dataSource: '',
    fileType: '',
  });
  const [useFilters, setUseFilters] = useState<boolean>(false);

  // Smart decryption helper
  const smartDecrypt = async (
    encryptedData: string | Uint8Array,
    progressCallback?: (progress: number) => void
  ): Promise<string | Uint8Array> => {
    try {
      const dataString =
        typeof encryptedData === 'string'
          ? encryptedData
          : encryptedData instanceof Uint8Array
          ? Buffer.from(encryptedData).toString('utf8')
          : String(encryptedData);

      if (
        dataString.includes('|METADATA_SEPARATOR|') &&
        dataString.includes('|CHUNK_SEPARATOR|')
      ) {
        const streamer = new StreamingEncryption();
        return await streamer.decryptFileStream(dataString, progressCallback || null);
      } else {
        return decryptFile(encryptedData);
      }
    } catch (error) {
      return decryptFile(encryptedData);
    }
  };

  const handleSearchSubmit = async () => {
    if (!searchQuery.trim() && !useFilters) {
      alert('Please enter a search query or use filters');
      return;
    }

    setIsSearching(true);
    try {
      const backendUrl = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

      let endpoint = '/search';
      let requestBody: Record<string, unknown> = {};
      
      if (useFilters && Object.values(filters).some(value => value !== '')) {
        if (searchQuery.trim()) {
          endpoint = '/search_with_filter';
          requestBody = {
            query: searchQuery.trim(),
            filters: buildFiltersObject(),
            n_results: 10
          };
        } else {
          endpoint = '/filter';
          requestBody = {
            filters: buildFiltersObject(),
            n_results: 10
          };
        }
      } else {
        requestBody = {
          query: searchQuery.trim()
        };
      }

      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Search API failed: ${response.statusText}`);
      }

      const result = await response.json();
      setSearchResults(result.results || result.data || result || []);
    } catch (error) {
      console.error('Search Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Search Error: ${errorMessage}`);
    } finally {
      setIsSearching(false);
    }
  };

  const buildFiltersObject = () => {
    const filterObj: Record<string, string> = {};
    
    if (filters.dataType) filterObj['dataType'] = filters.dataType;
    if (filters.gender) filterObj['gender'] = filters.gender;
    if (filters.dataSource) filterObj['dataSource'] = filters.dataSource;
    if (filters.ageRange) filterObj['ageRange'] = filters.ageRange;
    if (filters.fileType) filterObj['fileType'] = filters.fileType;
    
    return filterObj;
  };

  const resetFilters = () => {
    setFilters({
      dataType: '',
      gender: '',
      ageRange: '',
      dataSource: '',
      fileType: ''
    });
    setUseFilters(false);
  };

  const handlePurchaseAndDownload = async (result: SearchResult, index: number) => {
    const cid = result.cid || result.ipfsHash || result.hash;
    if (!cid) return;

    setPurchasing(prev => ({ ...prev, [index]: true }));
    
    try {
      const price = result.metadata?.price || await getDocumentPrice(cid);
      
      if (!price || parseFloat(price) <= 0) {
        alert('Unable to get document price');
        return;
      }

      const confirmed = window.confirm(`Purchase this document for ${price} ETH?`);
      if (!confirmed) return;

      const txHash = await purchaseDocument(cid, price);

      setDownloading(prev => ({ ...prev, [index]: true }));
      
      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${cid}`);
      const encryptedData = await response.text();
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        () => {}
      );
      
      let bytes: Uint8Array;
      if (decryptedData instanceof Uint8Array) {
        bytes = decryptedData;
      } else {
        bytes = new Uint8Array(atob(decryptedData).split('').map(char => char.charCodeAt(0)));
      }
      
      const blob = new Blob([bytes.buffer as ArrayBuffer], { type: result.metadata?.fileType || 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = result.metadata?.fileName || `document_${index + 1}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Purchase/Download Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Error: ${errorMessage}`);
    } finally {
      setPurchasing(prev => ({ ...prev, [index]: false }));
      setDownloading(prev => ({ ...prev, [index]: false }));
    }
  };

  const handlePreviewView = async (result: SearchResult, index: number) => {
    const previewHash = result.metadata?.previewHash;
    if (!previewHash) return;

    setPreviewLoading(true);
    setShowPreviewModal(true);
    
    try {
      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${previewHash}`);
      const encryptedData = await response.text();
      
      const decryptedData = await smartDecrypt(
        encryptedData,
        () => {}
      );
      
      let bytes: Uint8Array;
      if (decryptedData instanceof Uint8Array) {
        bytes = decryptedData;
      } else {
        bytes = new Uint8Array(atob(decryptedData).split('').map(char => char.charCodeAt(0)));
      }
      
      try {
        const XLSX = await import('xlsx');
        
        const workbook = XLSX.read(bytes, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as unknown[][];
        
        const headers = ((jsonData[0] || []) as unknown[]).map(h => String(h));
        const dataRows = jsonData.slice(1) || [];
        
        const previewInfo = {
          fileName: result.metadata?.fileName || `document_${index + 1}`,
          fileSize: bytes.length,
          dataType: result.metadata?.fileType || 'Excel file',
          message: 'Excel preview data loaded successfully. This is a 5% sample of the anonymized dataset.',
          headers: headers,
          data: dataRows,
          totalRows: dataRows.length,
          totalColumns: headers.length,
          sheetName: firstSheetName
        };
        
        setPreviewData(previewInfo);
        
      } catch (parseError) {
        console.error('Error parsing Excel data:', parseError);
        const errorMessage = parseError instanceof Error ? parseError.message : 'Unknown error';
        setPreviewData({
          fileName: result.metadata?.fileName || `document_${index + 1}`,
          fileSize: bytes.length,
          dataType: 'File',
          message: 'Preview loaded but could not parse Excel content. The file may be corrupted or in an unsupported format.',
          error: errorMessage
        });
      }
      
    } catch (error) {
      console.error('Preview View Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setPreviewData({
        error: `Error loading preview: ${errorMessage}`,
        fileName: result.metadata?.fileName || 'Unknown file'
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Enhanced Header with Gradient */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-blue-600 to-indigo-600 h-2"></div>
      
      {/* Header Section */}
      <div className="bg-white/90 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="relative flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-3 px-5 py-3 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-all duration-200 shadow-lg hover:shadow-xl backdrop-blur-sm z-10"
            >
              <ArrowLeft size={18} className="text-gray-600" />
              <span className="font-medium text-gray-700">Back to Main</span>
            </button>
            
            <div className="absolute left-0 right-0 text-center pointer-events-none">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Search Documents
              </h2>
              <p className="text-gray-600 text-base mt-1">Find medical documents using advanced search and filtering</p>
            </div>
            
            <div className="w-32"></div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* Search Card */}
        <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8 mb-8 border border-gray-100">
          {/* Search Input */}
          <div className="mb-6">
            <div className="flex items-center gap-3 w-full px-4 py-4 border border-gray-300 rounded-2xl bg-white shadow-sm focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent">
              <Search size={20} className="text-gray-400 flex-shrink-0" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter your search query..."
                className="flex-1 min-w-0 text-lg bg-transparent outline-none border-none placeholder-gray-400"
                onKeyPress={(e) => e.key === 'Enter' && handleSearchSubmit()}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <button
                onClick={handleSearchSubmit}
                disabled={isSearching}
                className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none font-semibold"
              >
                {isSearching ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Searching...
                  </>
                ) : (
                  <>
                    <Search size={20} />
                    Search Documents
                  </>
                )}
              </button>

              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center gap-3 px-6 py-4 border-2 rounded-2xl transition-all duration-200 font-medium ${
                  showFilters 
                    ? 'border-blue-500 bg-blue-50 text-blue-700' 
                    : 'border-gray-300 bg-white text-gray-700 hover:border-blue-400 hover:bg-blue-50'
                }`}
              >
                <Filter size={18} />
                {showFilters ? 'Hide Filters' : 'Show Filters'}
              </button>
            </div>
            
            {useFilters && (
              <div className="flex items-center gap-3 px-5 py-3 bg-green-50 border border-green-200 rounded-xl">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-semibold text-green-700">Filters Active</span>
                <button
                  onClick={resetFilters}
                  className="flex items-center gap-1.5 text-sm text-red-600 hover:text-red-700 font-semibold ml-2"
                >
                  <X size={15} />
                  <span>Clear</span>
                </button>
              </div>
            )}
          </div>
          
          {/* Advanced Filters Panel */}
          {showFilters && (
            <div className="mt-8 p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-2xl border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Filter size={20} className="text-blue-600" />
                Advanced Filters
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                {/* Data Type Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Building size={16} className="text-blue-600" />
                    Data Type
                  </label>
                  <select
                    value={filters.dataType}
                    onChange={(e) => {
                      setFilters({...filters, dataType: e.target.value});
                      setUseFilters(true);
                    }}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white transition-all duration-200"
                  >
                    <option value="">All Types</option>
                    <option value="Personal">Personal</option>
                    <option value="Institution">Institution</option>
                  </select>
                </div>
                
                {/* Gender Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <User size={16} className="text-purple-600" />
                    Gender
                  </label>
                  <select
                    value={filters.gender}
                    onChange={(e) => {
                      setFilters({...filters, gender: e.target.value});
                      setUseFilters(true);
                    }}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white transition-all duration-200"
                  >
                    <option value="">All Genders</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                
                {/* Data Source Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Building size={16} className="text-green-600" />
                    Data Source
                  </label>
                  <select
                    value={filters.dataSource}
                    onChange={(e) => {
                      setFilters({...filters, dataSource: e.target.value});
                      setUseFilters(true);
                    }}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white transition-all duration-200"
                  >
                    <option value="">All Sources</option>
                    <option value="Hospital">Hospital</option>
                    <option value="Clinic">Clinic</option>
                    <option value="Laboratory">Laboratory</option>
                    <option value="Research Institution">Research Institution</option>
                    <option value="Medical Device">Medical Device</option>
                    <option value="Electronic Health Record">Electronic Health Record</option>
                    <option value="Patient Self-Reported">Patient Self-Reported</option>
                    <option value="Insurance Claims">Insurance Claims</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                
                {/* File Type Filter */}
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <FileText size={16} className="text-orange-600" />
                    File Type
                  </label>
                  <select
                    value={filters.fileType}
                    onChange={(e) => {
                      setFilters({...filters, fileType: e.target.value});
                      setUseFilters(true);
                    }}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white transition-all duration-200"
                  >
                    <option value="">All Types</option>
                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">Excel (XLSX)</option>
                    <option value="application/vnd.ms-excel">Excel (XLS)</option>
                    <option value="image/jpeg">JPEG Image</option>
                    <option value="image/png">PNG Image</option>
                  </select>
                </div>
              </div>
              
              {/* Filter Action Buttons */}
              <div className="flex justify-end gap-3">
                <button
                  onClick={resetFilters}
                  className="px-6 py-3 text-gray-600 hover:text-gray-700 border border-gray-300 rounded-xl hover:bg-gray-50 transition-all duration-200 font-medium"
                >
                  Clear All Filters
                </button>
                <button
                  onClick={handleSearchSubmit}
                  disabled={isSearching}
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 transition-all duration-200 font-medium shadow-lg hover:shadow-xl"
                >
                  Apply Filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {searchResults && (
          <div className="space-y-6">
            {(() => {
              const resultsArray = Array.isArray(searchResults) 
                ? searchResults 
                : [];
              
              return (
                <>
                  {/* Results Header */}
                  <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6 border border-gray-100 shadow-lg">
                    <div className="flex items-center justify-between">
                      <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
                        <Eye size={24} className="text-blue-600" />
                        {useFilters ? 'Filtered Results' : 'Search Results'}
                      </h3>
                      <div className="flex items-center gap-4">
                        <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-xl font-semibold">
                          {resultsArray.length || 0} found
                        </span>
                        {useFilters && (
                          <span className="px-4 py-2 bg-green-100 text-green-700 rounded-xl font-medium text-sm flex items-center gap-2">
                            <Filter size={14} />
                            Filters applied
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {/* Results Grid */}
                  {resultsArray.length === 0 ? (
                    <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-12 text-center border border-gray-100">
                      <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                        <Search size={32} className="text-gray-400" />
                      </div>
                      <h3 className="text-xl font-semibold text-gray-800 mb-2">No Documents Found</h3>
                      <p className="text-gray-500 mb-6">No documents found matching your search criteria. Try adjusting your search terms or filters.</p>
                    </div>
                  ) : (
                    <div className="grid gap-6">
                      {resultsArray.map((result, index) => (
                        <div key={index} className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8 border border-gray-100 hover:shadow-2xl transition-all duration-200">
                          {/* Document Header */}
                          <div className="flex items-start justify-between mb-6">
                            <div className="flex-1">
                              <h4 className="text-xl font-bold text-gray-800 mb-3 flex items-center gap-3">
                                <FileText size={20} className="text-blue-600 flex-shrink-0" />
                                {(() => {
                                  const fullText = result.summary || result.description || result.content || result.document || '';
                                  const titleMatch = fullText.match(/^Dataset Title:\s*(.+?)$/m);
                                  return titleMatch ? titleMatch[1].trim() : (result.metadata?.fileName || result.title || result.fileName || `Document ${index + 1}`);
                                })()}
                              </h4>
                              
                              <p className="text-gray-600 leading-relaxed mb-4">
                                {(() => {
                                  const fullText = result.summary || result.description || result.content || result.document || '';
                                  const summaryMatch = fullText.match(/^Dataset Title:\s*.+?\n(.+?)(?:\nDisease Tags:|$)/s);
                                  if (summaryMatch) {
                                    return summaryMatch[1].trim();
                                  }
                                  return fullText || 'No summary available';
                                })()}
                              </p>
                            </div>
                            
                            {/* Price Badge */}
                            {result.metadata?.price && (
                              <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-4 py-2 rounded-xl font-bold text-lg shadow-lg">
                                {result.metadata.price} ETH
                              </div>
                            )}
                          </div>
                          
                          {/* Metadata Grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                            {result.metadata?.fileType && (
                              <div className="bg-blue-50 rounded-xl p-3">
                                <div className="text-xs font-medium text-blue-600 mb-1">File Type</div>
                                <div className="text-sm font-semibold text-blue-800">{result.metadata.fileType.split('/').pop()?.toUpperCase()}</div>
                              </div>
                            )}
                            {result.metadata?.fileSize && (
                              <div className="bg-purple-50 rounded-xl p-3">
                                <div className="text-xs font-medium text-purple-600 mb-1">Size</div>
                                <div className="text-sm font-semibold text-purple-800">{(result.metadata.fileSize / 1024).toFixed(1)} KB</div>
                              </div>
                            )}
                            {result.metadata?.uploadDate && (
                              <div className="bg-orange-50 rounded-xl p-3">
                                <div className="text-xs font-medium text-orange-600 mb-1 flex items-center gap-1">
                                  <Calendar size={12} />
                                  Uploaded
                                </div>
                                <div className="text-sm font-semibold text-orange-800">{new Date(result.metadata.uploadDate).toLocaleDateString()}</div>
                              </div>
                            )}
                            {result.metadata?.dataSource && (
                              <div className="bg-green-50 rounded-xl p-3">
                                <div className="text-xs font-medium text-green-600 mb-1">Source</div>
                                <div className="text-sm font-semibold text-green-800">{result.metadata.dataSource}</div>
                              </div>
                            )}
                          </div>
                          
                          {/* Additional Metadata */}
                          <div className="flex flex-wrap gap-2 mb-6">
                            {result.metadata?.gender && (
                              <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-xs font-medium">
                                {result.metadata.gender}
                              </span>
                            )}
                            {result.metadata?.dataType && (
                              <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs font-medium">
                                {result.metadata.dataType}
                              </span>
                            )}
                            {result.metadata?.previewHash && (
                              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                                Preview Available
                              </span>
                            )}
                          </div>
                          
                          {/* Action Buttons */}
                          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200">
                            {/* Preview Button for Excel files */}
                            {result.metadata?.previewHash && result.metadata?.fileType?.includes('spreadsheet') && (
                              <button
                                onClick={() => handlePreviewView(result, index)}
                                disabled={previewLoading}
                                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-105 disabled:transform-none font-medium"
                              >
                                {previewLoading ? (
                                  <>
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    Loading Preview...
                                  </>
                                ) : (
                                  <>
                                    <Eye className="w-4 h-4" />
                                    View Preview (5%)
                                  </>
                                )}
                              </button>
                            )}
                            
                            {/* Purchase & Download Button */}
                            {(result.cid || result.ipfsHash || result.hash) && (
                              <button
                                onClick={() => handlePurchaseAndDownload(result, index)}
                                disabled={purchasing[index] || downloading[index]}
                                className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-2xl hover:from-green-700 hover:to-emerald-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none font-semibold"
                              >
                                {purchasing[index] ? (
                                  <>
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    Purchasing...
                                  </>
                                ) : downloading[index] ? (
                                  <>
                                    <Download size={20} />
                                    Downloading...
                                  </>
                                ) : (
                                  <>
                                    <ShoppingCart size={20} />
                                    Purchase & Download
                                    <span className="text-green-100">({result.metadata?.price || '...'} ETH)</span>
                                  </>
                                )}
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        )}

        {/* Preview Modal */}
        {showPreviewModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
                <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
                  <Eye size={24} className="text-blue-600" />
                  Excel Preview (5% Sample)
                </h3>
                <button
                  onClick={() => {
                    setShowPreviewModal(false);
                    setPreviewData(null);
                  }}
                  className="p-2 hover:bg-gray-100 rounded-full transition-colors duration-200"
                >
                  <X size={24} className="text-gray-500 hover:text-gray-700" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
                {previewLoading ? (
                  <div className="flex flex-col items-center justify-center py-12">
                    <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p className="text-gray-600 font-medium">Loading preview data...</p>
                  </div>
                ) : previewData?.error ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <X size={32} className="text-red-500" />
                    </div>
                    <h4 className="text-xl font-semibold text-red-600 mb-2">Error Loading Preview</h4>
                    <p className="text-gray-600">{previewData.error}</p>
                  </div>
                ) : previewData ? (
                  <div className="space-y-6">
                    {/* File Information */}
                    <div className="bg-gradient-to-r from-gray-50 to-blue-50 rounded-2xl p-6">
                      <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                        <FileText size={20} className="text-blue-600" />
                        Dataset Overview
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <span className="text-sm font-medium text-gray-600">File Name:</span>
                          <p className="text-gray-800 font-semibold break-all">{previewData.fileName || 'Unknown'}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-600">File Size:</span>
                          <p className="text-gray-800 font-semibold">
                            {previewData.fileSize ? `${(previewData.fileSize / 1024).toFixed(1)} KB` : 'Unknown'}
                          </p>
                        </div>
                        {previewData.totalRows && (
                          <div>
                            <span className="text-sm font-medium text-gray-600">Rows:</span>
                            <p className="text-gray-800 font-semibold">{previewData.totalRows}</p>
                          </div>
                        )}
                        {previewData.totalColumns && (
                          <div>
                            <span className="text-sm font-medium text-gray-600">Columns:</span>
                            <p className="text-gray-800 font-semibold">{previewData.totalColumns}</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Excel Data Table */}
                    {previewData.headers && previewData.data ? (
                      <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden">
                        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-6 py-4 border-b border-gray-200">
                          <h4 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                            Preview Data
                            {previewData.sheetName && (
                              <span className="text-sm font-normal text-gray-600">({previewData.sheetName})</span>
                            )}
                          </h4>
                          <p className="text-sm text-gray-600 mt-1">
                            Showing all {previewData.totalRows} rows (5% sample of the full dataset)
                          </p>
                        </div>
                        
                        <div className="overflow-x-auto max-h-96">
                          <table className="min-w-full">
                            <thead className="bg-gray-50 sticky top-0">
                              <tr>
                                <th className="px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200 bg-gray-100">
                                  #
                                </th>
                                {previewData.headers.map((header, idx) => (
                                  <th key={idx} className="px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200 bg-gray-100 min-w-[120px]">
                                    {header || `Column ${idx + 1}`}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            
                            <tbody className="bg-white divide-y divide-gray-200">
                              {previewData.data.map((row, rowIdx) => (
                                <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                  <td className="px-3 py-2 text-sm text-gray-500 border-r border-gray-200 bg-gray-50 font-medium">
                                    {rowIdx + 1}
                                  </td>
                                  {previewData.headers!.map((_, colIdx) => (
                                    <td key={colIdx} className="px-3 py-2 text-sm text-gray-900 border-r border-gray-200 max-w-[200px] truncate">
                                      {(row as any[])[colIdx] !== undefined && (row as any[])[colIdx] !== null ? 
                                        String((row as any[])[colIdx]) : 
                                        <span className="text-gray-400 italic">—</span>
                                      }
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                            <Eye size={16} className="text-white" />
                          </div>
                          <div>
                            <h5 className="font-semibold text-blue-800 mb-2">Preview Information</h5>
                            <p className="text-blue-700">{previewData.message}</p>
                            {previewData.error && (
                              <p className="text-red-600 mt-2 text-sm">Error: {previewData.error}</p>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Preview Note */}
                    <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6">
                      <div className="flex items-start gap-3">
                        <div className="w-8 h-8 bg-yellow-500 rounded-full flex items-center justify-center flex-shrink-0">
                          <FileText size={16} className="text-white" />
                        </div>
                        <div>
                          <h5 className="font-semibold text-yellow-800 mb-2">About This Preview</h5>
                          <ul className="text-yellow-700 space-y-1 text-sm">
                            <li>• This preview contains the first 5% of rows from the anonymized dataset</li>
                            <li>• All personal health information (PHI) has been anonymized</li>
                            <li>• The preview shows the actual data structure and quality you&apos;ll receive</li>
                            <li>• Full dataset purchase includes all remaining data with the same anonymization quality</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex justify-between items-center pt-4 border-t border-gray-200">
                      <button
                        onClick={() => {
                          setShowPreviewModal(false);
                          setPreviewData(null);
                        }}
                        className="px-6 py-3 text-gray-600 hover:text-gray-700 border border-gray-300 rounded-xl hover:bg-gray-50 transition-all duration-200 font-medium"
                      >
                        Close Preview
                      </button>
                      
                      <button
                        onClick={() => {
                          setShowPreviewModal(false);
                          setPreviewData(null);
                        }}
                        className="px-8 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl hover:from-green-700 hover:to-emerald-700 transition-all duration-200 shadow-md hover:shadow-lg font-medium"
                      >
                        Purchase Full Dataset
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <FileText size={32} className="text-gray-400" />
                    </div>
                    <p className="text-gray-500">No preview data available</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
