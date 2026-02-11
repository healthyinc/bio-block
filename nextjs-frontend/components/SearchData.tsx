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
      console.warn('Streaming decryption failed, falling back to traditional:', error);
      return decryptFile(encryptedData);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim() && !useFilters) {
      alert('Please enter a search query or use filters');
      return;
    }

    setIsSearching(true);
    setSearchResults(null);

    try {
      const pythonBackendUrl =
        process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

      let url: string;
      let body: Record<string, unknown> | null = null;

      if (useFilters) {
        const filterObject = convertFiltersToBackendFormat();
        if (searchQuery.trim()) {
          url = `${pythonBackendUrl}/search_with_filter`;
          body = {
            query: searchQuery.trim(),
            filters: filterObject,
          };
        } else {
          url = `${pythonBackendUrl}/filter`;
          body = { filters: filterObject };
        }
      } else {
        url = `${pythonBackendUrl}/search?query=${encodeURIComponent(
          searchQuery.trim()
        )}`;
      }

      const fetchOptions: RequestInit = {
        method: body ? 'POST' : 'GET',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      };

      const response = await fetch(url, fetchOptions);

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      const results = data.results || data.data || data || [];
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Search failed: ${errorMessage}`);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const convertFiltersToBackendFormat = (): Record<string, string> => {
    const backendFilters: Record<string, string> = {};
    if (filters.dataType) backendFilters.dataType = filters.dataType;
    if (filters.gender) backendFilters.gender = filters.gender;
    if (filters.dataSource) backendFilters.dataSource = filters.dataSource;
    if (filters.fileType) backendFilters.fileType = filters.fileType;
    return backendFilters;
  };

  const clearFilters = () => {
    setFilters({
      dataType: '',
      gender: '',
      ageRange: '',
      dataSource: '',
      fileType: '',
    });
    setUseFilters(false);
  };

  const handlePurchaseAndDownload = async (result: SearchResult, index: number) => {
    const hash = result.cid || result.ipfsHash || result.hash;
    if (!hash) {
      alert('Document hash not found');
      return;
    }

    setPurchasing((prev) => ({ ...prev, [index]: true }));

    try {
      const price = await getDocumentPrice(hash);
      const confirmed = window.confirm(
        `Purchase this document for ${price} ETH?\n\nThis will download the file after purchase.`
      );

      if (!confirmed) {
        setPurchasing((prev) => ({ ...prev, [index]: false }));
        return;
      }

      const txHash = await purchaseDocument(hash, price);
      setDownloading((prev) => ({ ...prev, [index]: true }));

      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${hash}`);
      if (!response.ok) {
        throw new Error('Failed to fetch document from IPFS');
      }

      const encryptedData = await response.text();
      const decryptedData = await smartDecrypt(encryptedData);

      let bytes: Uint8Array;
      if (typeof decryptedData === 'string') {
        bytes = new Uint8Array(
          atob(decryptedData)
            .split('')
            .map((char) => char.charCodeAt(0))
        );
      } else {
        bytes = decryptedData;
      }

      // Create a proper ArrayBuffer for the Blob
      const arrayBuffer = bytes.slice().buffer as ArrayBuffer;
      const blob = new Blob([arrayBuffer], {
        type:
          result.metadata?.fileType ||
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = result.metadata?.fileName || result.fileName || hash;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      alert('Purchase successful! File downloaded.');
    } catch (error) {
      console.error('Purchase/Download error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Failed: ${errorMessage}`);
    } finally {
      setPurchasing((prev) => ({ ...prev, [index]: false }));
      setDownloading((prev) => ({ ...prev, [index]: false }));
    }
  };

  const handleViewPreview = async (result: SearchResult) => {
    const previewHash = result.metadata?.previewHash;
    if (!previewHash) {
      alert('No preview available for this document');
      return;
    }

    setShowPreviewModal(true);
    setPreviewLoading(true);
    setPreviewData(null);

    try {
      const response = await fetch(
        `https://gateway.pinata.cloud/ipfs/${previewHash}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch preview');
      }

      const encryptedPreview = await response.text();
      const decryptedPreview = await smartDecrypt(encryptedPreview);

      let bytes: Uint8Array;
      if (typeof decryptedPreview === 'string') {
        bytes = new Uint8Array(
          atob(decryptedPreview)
            .split('')
            .map((char) => char.charCodeAt(0))
        );
      } else {
        bytes = decryptedPreview;
      }

      // Dynamic import of XLSX
      const XLSX = await import('xlsx');
      const workbook = XLSX.read(bytes, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      const headers = (jsonData[0] as string[]) || [];
      const dataRows = jsonData.slice(1) as unknown[][];

      setPreviewData({
        fileName: result.metadata?.fileName || 'Unknown',
        fileSize: result.metadata?.fileSize,
        dataType: result.metadata?.dataType,
        headers: headers,
        data: dataRows,
        totalRows: dataRows.length,
        totalColumns: headers.length,
        sheetName: firstSheetName,
      });
    } catch (error) {
      console.error('Preview error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setPreviewData({
        error: `Failed to load preview: ${errorMessage}`,
      });
    } finally {
      setPreviewLoading(false);
    }
  };

  const extractTitle = (result: SearchResult): string => {
    if (result.title) return result.title;
    if (result.summary) {
      const lines = result.summary.split('\n');
      return lines[0].substring(0, 60) + (lines[0].length > 60 ? '...' : '');
    }
    if (result.metadata?.fileName) return result.metadata.fileName;
    return 'Untitled Document';
  };

  const extractDescription = (result: SearchResult): string => {
    return (
      result.description ||
      result.summary ||
      result.content ||
      result.document ||
      'No description available'
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Sticky Header */}
      <div className="sticky top-0 z-40 bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors backdrop-blur-sm"
            >
              <ArrowLeft size={20} />
              <span className="font-medium">Back</span>
            </button>
            <h1 className="text-2xl md:text-3xl font-bold">Search Documents</h1>
            <div className="w-24"></div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Search Card */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-8">
          <div className="flex gap-4 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search by disease, symptoms, or data type..."
                className="w-full pl-12 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
            >
              {isSearching ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Searching...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Search
                </>
              )}
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-6 py-3 rounded-lg transition-colors flex items-center gap-2 font-medium ${
                showFilters
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Filter size={20} />
              {showFilters ? 'Hide Filters' : 'Show Filters'}
            </button>
          </div>

          {useFilters && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
              <p className="text-sm text-blue-800 font-medium">
                ✓ Filters Active - Search results will be filtered
              </p>
            </div>
          )}

          {/* Advanced Filters */}
          {showFilters && (
            <div className="border-t pt-6 space-y-4">
              <h3 className="font-semibold text-gray-800 mb-4">Advanced Filters</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Data Type
                  </label>
                  <select
                    value={filters.dataType}
                    onChange={(e) =>
                      setFilters({ ...filters, dataType: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All</option>
                    <option value="Personal">Personal</option>
                    <option value="Institution">Institution</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender
                  </label>
                  <select
                    value={filters.gender}
                    onChange={(e) =>
                      setFilters({ ...filters, gender: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Data Source
                  </label>
                  <select
                    value={filters.dataSource}
                    onChange={(e) =>
                      setFilters({ ...filters, dataSource: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All</option>
                    <option value="Hospital">Hospital</option>
                    <option value="Clinic">Clinic</option>
                    <option value="Laboratory">Laboratory</option>
                    <option value="Research Institution">Research Institution</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    File Type
                  </label>
                  <select
                    value={filters.fileType}
                    onChange={(e) =>
                      setFilters({ ...filters, fileType: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">All</option>
                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">
                      Excel (XLSX)
                    </option>
                    <option value="application/vnd.ms-excel">Excel (XLS)</option>
                    <option value="image/jpeg">JPEG Image</option>
                    <option value="image/png">PNG Image</option>
                  </select>
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={clearFilters}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Clear All Filters
                </button>
                <button
                  onClick={() => setUseFilters(true)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Apply Filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {searchResults !== null && (
          <div>
            <div className="flex items-center gap-3 mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Search Results</h2>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                {searchResults.length}
              </span>
            </div>

            {searchResults.length === 0 ? (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <FileText size={64} className="mx-auto text-gray-300 mb-4" />
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  No Results Found
                </h3>
                <p className="text-gray-600">
                  Try adjusting your search query or filters
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {searchResults.map((result, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow p-6"
                  >
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="text-lg font-bold text-gray-800">
                        {extractTitle(result)}
                      </h3>
                      <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                        {result.metadata?.price || '0.001'} ETH
                      </span>
                    </div>

                    <p className="text-gray-600 mb-4 line-clamp-3">
                      {extractDescription(result)}
                    </p>

                    <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
                      {result.metadata?.fileType && (
                        <div className="flex items-center gap-2 text-gray-600">
                          <FileText size={16} />
                          <span>
                            {result.metadata.fileType.includes('spreadsheet')
                              ? 'Excel'
                              : 'File'}
                          </span>
                        </div>
                      )}
                      {result.metadata?.uploadDate && (
                        <div className="flex items-center gap-2 text-gray-600">
                          <Calendar size={16} />
                          <span>
                            {new Date(result.metadata.uploadDate).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                      {result.metadata?.gender && (
                        <div className="flex items-center gap-2 text-gray-600">
                          <User size={16} />
                          <span>{result.metadata.gender}</span>
                        </div>
                      )}
                      {result.metadata?.dataSource && (
                        <div className="flex items-center gap-2 text-gray-600">
                          <Building size={16} />
                          <span>{result.metadata.dataSource}</span>
                        </div>
                      )}
                    </div>

                    <div className="flex gap-2 mb-4 flex-wrap">
                      {result.metadata?.dataType && (
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                          {result.metadata.dataType}
                        </span>
                      )}
                      {result.metadata?.previewHash && (
                        <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">
                          Preview Available
                        </span>
                      )}
                    </div>

                    <div className="flex gap-3">
                      {result.metadata?.previewHash && (
                        <button
                          onClick={() => handleViewPreview(result)}
                          className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
                        >
                          <Eye size={18} />
                          View Preview
                        </button>
                      )}
                      <button
                        onClick={() => handlePurchaseAndDownload(result, index)}
                        disabled={purchasing[index] || downloading[index]}
                        className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                      >
                        {purchasing[index] ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                            Purchasing...
                          </>
                        ) : downloading[index] ? (
                          <>
                            <Download size={18} />
                            Downloading...
                          </>
                        ) : (
                          <>
                            <ShoppingCart size={18} />
                            Purchase & Download
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Preview Modal */}
      {showPreviewModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-2xl font-bold text-gray-800">Document Preview</h2>
              <button
                onClick={() => setShowPreviewModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 overflow-y-auto max-h-[calc(90vh-180px)]">
              {previewLoading ? (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading preview...</p>
                </div>
              ) : previewData?.error ? (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                  <p className="text-red-600">{previewData.error}</p>
                </div>
              ) : previewData ? (
                <>
                  <div className="bg-gray-50 rounded-lg p-4 mb-6">
                    <h3 className="font-semibold mb-2">Dataset Overview</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">File Name:</span>
                        <p className="font-medium">{previewData.fileName}</p>
                      </div>
                      {previewData.totalRows !== undefined && (
                        <div>
                          <span className="text-gray-600">Rows:</span>
                          <p className="font-medium">{previewData.totalRows}</p>
                        </div>
                      )}
                      {previewData.totalColumns !== undefined && (
                        <div>
                          <span className="text-gray-600">Columns:</span>
                          <p className="font-medium">{previewData.totalColumns}</p>
                        </div>
                      )}
                      {previewData.fileSize && (
                        <div>
                          <span className="text-gray-600">Size:</span>
                          <p className="font-medium">
                            {(previewData.fileSize / 1024).toFixed(2)} KB
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  {previewData.headers && previewData.data && (
                    <div className="overflow-x-auto">
                      <table className="min-w-full border-collapse">
                        <thead className="sticky top-0 bg-gray-100">
                          <tr>
                            {previewData.headers.map((header, idx) => (
                              <th
                                key={idx}
                                className="border px-4 py-2 text-left text-sm font-medium"
                              >
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {previewData.data.slice(0, 20).map((row, rowIdx) => (
                            <tr key={rowIdx} className="hover:bg-gray-50">
                              {(row as unknown[]).map((cell, cellIdx) => (
                                <td key={cellIdx} className="border px-4 py-2 text-sm">
                                  {String(cell)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-6">
                    <p className="text-sm text-yellow-800">
                      ℹ️ This is an anonymized preview. Purchase the full dataset to
                      access all data.
                    </p>
                  </div>
                </>
              ) : null}
            </div>

            <div className="p-6 border-t flex gap-4">
              <button
                onClick={() => setShowPreviewModal(false)}
                className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
