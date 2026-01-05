"use client";

import React, { useState, useEffect } from 'react';
import {
  Upload,
  Wallet,
  ArrowLeft,
  Shield,
  Database,
  CheckCircle,
  X,
  Clock,
  Check,
} from 'lucide-react';
import { storeDocumentHash } from '../lib/contractService';
import { encryptFile } from '../lib/encryptionUtils';
import StreamingEncryption from '../lib/streamingEncryption';

// Types
interface UploadDataProps {
  onBack: () => void;
  isWalletConnected: boolean;
  walletAddress: string;
  onWalletConnect: () => void;
}

type PreviewType = 'image' | 'spreadsheet' | 'pdf' | 'dicom' | null;

interface SpreadsheetPreviewData {
  fileName: string;
  message?: string;
  headers: string[];
  data: string[][];
  totalRows: number;
  previewRows: number;
}

interface UploadStep {
  name: string;
  completed: boolean;
  error: boolean;
}

interface UploadProgress {
  steps: UploadStep[];
  ipfsHash: string;
  transactionHash: string;
  price: string;
  isComplete: boolean;
  hasError: boolean;
  errorMessage: string;
}

interface AnonymizeResult {
  mainFile: File;
  previewFile: File | null;
}

interface IPFSUploadResult {
  success: boolean;
  ipfsHash?: string;
  error?: string;
}

export default function UploadData({
  onBack,
  isWalletConnected,
  walletAddress,
  onWalletConnect,
}: UploadDataProps) {
  // State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetTitle, setDatasetTitle] = useState<string>('');
  const [summary, setSummary] = useState<string>('');
  const [diseaseTags, setDiseaseTags] = useState<string[]>([]);
  const [dataType, setDataType] = useState<string>('');
  const [gender, setGender] = useState<string>('');
  const [dataSource, setDataSource] = useState<string>('');
  const [price, setPrice] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewType, setPreviewType] = useState<PreviewType>(null);
  const [previewData, setPreviewData] = useState<SpreadsheetPreviewData | null>(
    null
  );
  const [isGeneratingPreview, setIsGeneratingPreview] = useState<boolean>(false);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [encryptionProgress, setEncryptionProgress] = useState<number>(0);
  const [isStreamingEncryption, setIsStreamingEncryption] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    steps: [
      { name: 'Preparing file...', completed: false, error: false },
      {
        name: 'Anonymizing data (if Excel/Image)...',
        completed: false,
        error: false,
      },
      { name: 'Encrypting file...', completed: false, error: false },
      { name: 'Uploading to IPFS...', completed: false, error: false },
      { name: 'Storing on blockchain...', completed: false, error: false },
      { name: 'Saving metadata...', completed: false, error: false },
    ],
    ipfsHash: '',
    transactionHash: '',
    price: '',
    isComplete: false,
    hasError: false,
    errorMessage: '',
  });

  const diseaseOptions = [
    'Cancer',
    'Diabetes',
    'Heart Disease',
    'Hypertension',
    'Stroke',
    'Asthma',
    'COPD (Chronic Obstructive Pulmonary Disease)',
    'Kidney Disease',
    'Liver Disease',
    'Arthritis',
    'Osteoporosis',
    "Alzheimer's Disease",
    "Parkinson's Disease",
    'Multiple Sclerosis',
    'Epilepsy',
    'Depression',
    'Anxiety Disorders',
    'Bipolar Disorder',
    'Schizophrenia',
    'Autism Spectrum Disorder',
    'ADHD (Attention Deficit Hyperactivity Disorder)',
    'Obesity',
    'Eating Disorders',
    'Sleep Disorders',
    'Migraine',
    'Thyroid Disorders',
    'Autoimmune Diseases',
    'Infectious Diseases',
    'Respiratory Infections',
    'Gastrointestinal Disorders',
    'Skin Conditions',
    'Eye Diseases',
    'Hearing Loss',
    'Blood Disorders',
    'Bone Fractures',
    'Sports Injuries',
    'Pregnancy Related',
    'Pediatric Conditions',
    'Geriatric Conditions',
    'Rare Diseases',
    'Genetic Disorders',
    'Surgical Procedures',
    'Emergency Medicine',
    'Rehabilitation',
    'Preventive Care',
    'Mental Health',
    'Substance Abuse',
    'Other',
  ];

  // Cleanup preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      setPreviewUrl(null);
      setPreviewType(null);
      setPreviewData(null);
    };
  }, []);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    // Memory leak prevention
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }

    setPreviewType(null);
    setPreviewData(null);
    setSelectedFile(file || null);

    if (!file) return;

    const jsBackendUrl =
      process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';
    setIsGeneratingPreview(true);

    try {
      const fileExtension = file.name.split('.').pop()?.toLowerCase();
      const isSpreadsheet =
        file.type.includes('spreadsheet') ||
        file.type.includes('csv') ||
        ['xlsx', 'xls', 'csv', 'ods', 'tsv', 'xlsm', 'xlsb'].includes(
          fileExtension || ''
        );
      const isPdf = file.type === 'application/pdf' || fileExtension === 'pdf';
      const isDicom = fileExtension === 'dcm' || fileExtension === 'dicom';

      if (file.type.startsWith('image/')) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/image`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Preview generation failed');

        const imageBlob = await response.blob();
        const objectUrl = URL.createObjectURL(imageBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('image');
      } else if (isSpreadsheet) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(
          `${jsBackendUrl}/api/preview/spreadsheet`,
          {
            method: 'POST',
            body: formData,
          }
        );

        if (!response.ok) throw new Error('Preview generation failed');

        const spreadsheetData: SpreadsheetPreviewData = await response.json();
        setPreviewData(spreadsheetData);
        setPreviewType('spreadsheet');
      } else if (isPdf) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/pdf`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Preview generation failed');

        const pdfBlob = await response.blob();
        const objectUrl = URL.createObjectURL(pdfBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('pdf');
      } else if (isDicom) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/dicom`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Preview generation failed');

        const imageBlob = await response.blob();
        const objectUrl = URL.createObjectURL(imageBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('dicom');
      }
    } catch (error) {
      console.error('Error generating preview:', error);
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      alert(`Preview generation failed: ${errorMessage}`);
      setPreviewUrl(null);
      setPreviewType(null);
      setPreviewData(null);
    } finally {
      setIsGeneratingPreview(false);
    }
  };

  const resetProgress = () => {
    setUploadProgress({
      steps: [
        { name: 'Preparing file...', completed: false, error: false },
        {
          name: 'Anonymizing data (if Excel/Image)...',
          completed: false,
          error: false,
        },
        { name: 'Encrypting file...', completed: false, error: false },
        { name: 'Uploading to IPFS...', completed: false, error: false },
        { name: 'Storing on blockchain...', completed: false, error: false },
        { name: 'Saving metadata...', completed: false, error: false },
      ],
      ipfsHash: '',
      transactionHash: '',
      price: '',
      isComplete: false,
      hasError: false,
      errorMessage: '',
    });
    setCurrentStep(0);
  };

  const updateStep = (
    stepIndex: number,
    completed: boolean = false,
    error: boolean = false
  ) => {
    setUploadProgress((prev) => ({
      ...prev,
      steps: prev.steps.map((step, index) =>
        index === stepIndex ? { ...step, completed, error } : step
      ),
    }));

    if (!completed && !error) {
      setCurrentStep(stepIndex);
    } else if (completed && !error) {
      setCurrentStep(stepIndex + 1);
    }
  };

  const setError = (message: string) => {
    setUploadProgress((prev) => ({
      ...prev,
      hasError: true,
      errorMessage: message,
    }));
  };

  const closeModal = () => {
    setShowModal(false);
    if (uploadProgress.isComplete) {
      setSelectedFile(null);
      setDatasetTitle('');
      setSummary('');
      setDiseaseTags([]);
      setDataType('');
      setGender('');
      setDataSource('');
      setPrice('');
    }
    resetProgress();
  };

  const anonymizeFile = async (file: File): Promise<AnonymizeResult> => {
    const formData = new FormData();
    formData.append('file', file);

    if (dataType === 'Personal') {
      formData.append('walletAddress', walletAddress);
    }

    const shouldGeneratePreview = file.name.match(/\.(xlsx|xls)$/i);
    if (shouldGeneratePreview) {
      formData.append('generatePreview', 'true');
    }

    let backendUrl: string, endpoint: string;

    if (file.type.startsWith('image/')) {
      backendUrl =
        process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';
      endpoint = '/anonymize_image';
    } else if (file.name.match(/\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i)) {
      backendUrl =
        process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';
      endpoint = '/anonymize';
    } else {
      throw new Error(
        'File type not supported for anonymization. Only Excel and image files are supported.'
      );
    }

    const response = await fetch(`${backendUrl}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || errorData.error || 'File anonymization failed'
      );
    }

    if (
      shouldGeneratePreview &&
      response.headers.get('content-type')?.includes('application/json')
    ) {
      const jsonResponse = await response.json();

      const mainFileBuffer = Uint8Array.from(
        atob(jsonResponse.files.main.buffer),
        (c) => c.charCodeAt(0)
      );
      const mainFile = new File([mainFileBuffer], jsonResponse.files.main.filename, {
        type: jsonResponse.files.main.contentType,
      });

      const previewFileBuffer = Uint8Array.from(
        atob(jsonResponse.files.preview.buffer),
        (c) => c.charCodeAt(0)
      );
      const previewFile = new File(
        [previewFileBuffer],
        jsonResponse.files.preview.filename,
        {
          type: jsonResponse.files.preview.contentType,
        }
      );

      return { mainFile, previewFile };
    } else {
      const blob = await response.blob();
      const mainFile = new File([blob], `anonymized_${file.name}`, {
        type: file.type,
      });
      return { mainFile, previewFile: null };
    }
  };

  const uploadToIPFSViaBackend = async (
    encryptedData: Blob | Uint8Array,
    fileName: string
  ): Promise<IPFSUploadResult> => {
    const formData = new FormData();
    
    // Convert Uint8Array to Blob properly
    const blobData = encryptedData instanceof Blob 
      ? encryptedData 
      : new Blob([encryptedData.buffer as ArrayBuffer], { type: 'application/octet-stream' });
    
    formData.append('encryptedFile', blobData);
    formData.append('fileName', fileName);

    const backendUrl =
      process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';

    const response = await fetch(`${backendUrl}/ipfs/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || 'IPFS upload failed');
    }

    const result: IPFSUploadResult = await response.json();
    return result;
  };

  const handleUpload = async () => {
    if (!isWalletConnected) {
      alert('Please connect your wallet first');
      return;
    }

    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }

    if (!datasetTitle.trim()) {
      alert('Please provide a dataset title');
      return;
    }

    if (!summary.trim()) {
      alert('Please provide a description of your document');
      return;
    }

    if (!diseaseTags.length) {
      alert('Please select at least one disease tag');
      return;
    }

    if (!dataType) {
      alert('Please select data type');
      return;
    }

    if (!dataSource) {
      alert('Please select data source');
      return;
    }

    if (!price || parseFloat(price) <= 0) {
      alert('Please enter a valid price in ETH');
      return;
    }

    setIsUploading(true);
    setShowModal(true);
    resetProgress();

    try {
      // Step 1: Preparing file
      updateStep(0);
      await new Promise((resolve) => setTimeout(resolve, 500));
      let fileToUpload = selectedFile;
      let previewFile: File | null = null;
      updateStep(0, true);

      // Step 2: Anonymizing
      updateStep(1);
      const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();
      const isSpreadsheet = selectedFile.name.match(
        /\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i
      );
      const isImage = selectedFile.type.startsWith('image/');
      const isPdf =
        selectedFile.type === 'application/pdf' || fileExtension === 'pdf';
      const isDicom = fileExtension === 'dcm' || fileExtension === 'dicom';

      if (isSpreadsheet || isImage) {
        try {
          const result = await anonymizeFile(selectedFile);
          fileToUpload = result.mainFile;
          previewFile = result.previewFile;
          updateStep(1, true, false);
        } catch (error) {
          updateStep(1, false, true);
          const errorMessage =
            error instanceof Error ? error.message : 'Unknown error';
          setError(`Anonymization failed: ${errorMessage}`);
          return;
        }
      } else if (isPdf || isDicom) {
        fileToUpload = selectedFile;
        previewFile = null;
        updateStep(1, true, false);
      } else {
        updateStep(1, false, true);
        setError('File type not supported for upload.');
        return;
      }

      // Step 3: Encrypting
      updateStep(2);
      try {
        const streamer = new StreamingEncryption();
        const shouldUseStreaming = streamer.shouldUseStreaming(fileToUpload.size);

        let encryptedFile: Blob | Uint8Array;

        if (shouldUseStreaming) {
          setIsStreamingEncryption(true);
          setEncryptionProgress(0);

          encryptedFile = await streamer.encryptFileStream(
            fileToUpload,
            (progress) => {
              setEncryptionProgress(progress);
            }
          );

          setIsStreamingEncryption(false);
        } else {
          const fileBuffer = await fileToUpload.arrayBuffer();
          encryptedFile = encryptFile(new Uint8Array(fileBuffer));
        }

        updateStep(2, true, false);

        // Step 4: Uploading to IPFS
        updateStep(3);
        const result = await uploadToIPFSViaBackend(
          encryptedFile,
          fileToUpload.name
        );
        if (!result.success) {
          updateStep(3, false, true);
          setError(`IPFS upload failed: ${result.error}`);
          return;
        }
        updateStep(3, true, false);

        setUploadProgress((prev) => ({
          ...prev,
          ipfsHash: result.ipfsHash || '',
        }));

        // Step 5: Storing on blockchain
        updateStep(4);
        const txHash = await storeDocumentHash(result.ipfsHash || '', price);
        updateStep(4, true, false);

        setUploadProgress((prev) => ({
          ...prev,
          transactionHash: txHash,
          price: price,
        }));

        // Step 6: Saving metadata
        updateStep(5);

        let previewHash: string | null = null;
        if (previewFile) {
          try {
            const previewStreamer = new StreamingEncryption();
            const shouldUseStreamingForPreview =
              previewStreamer.shouldUseStreaming(previewFile.size);

            let encryptedPreview: Blob | Uint8Array;

            if (shouldUseStreamingForPreview) {
              encryptedPreview = await previewStreamer.encryptFileStream(
                previewFile
              );
            } else {
              const previewBuffer = await previewFile.arrayBuffer();
              encryptedPreview = encryptFile(new Uint8Array(previewBuffer));
            }

            const previewResult = await uploadToIPFSViaBackend(
              encryptedPreview,
              previewFile.name
            );
            if (previewResult.success) {
              previewHash = previewResult.ipfsHash || null;
            }
          } catch (error) {
            // Preview upload failed, continue without preview
          }
        }

        const metadata = {
          fileName: fileToUpload.name,
          fileSize: fileToUpload.size,
          fileType: fileToUpload.type,
          uploadDate: new Date().toISOString(),
          walletAddress: walletAddress,
          transactionHash: txHash,
          encrypted: true,
          price: price,
          dataType: dataType,
          gender: gender,
          dataSource: dataSource,
          datasetTitle: datasetTitle,
          summary: summary,
          ...(previewHash && { previewHash: previewHash }),
          ...(diseaseTags.length && { disease_tags: diseaseTags.join(', ') }),
        };

        const pythonBackendUrl =
          process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';
        const storeResponse = await fetch(`${pythonBackendUrl}/store`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ipfs_hash: result.ipfsHash,
            metadata: metadata,
          }),
        });

        if (!storeResponse.ok) {
          throw new Error('Metadata storage failed');
        }

        updateStep(5, true, false);

        setUploadProgress((prev) => ({
          ...prev,
          isComplete: true,
        }));
      } catch (error) {
        updateStep(2, false, true);
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown error';
        setError(`Encryption failed: ${errorMessage}`);
        return;
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      setError(`Upload failed: ${errorMessage}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleTagChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    if (value && !diseaseTags.includes(value)) {
      setDiseaseTags([...diseaseTags, value]);
    }
  };

  const removeTag = (tagToRemove: string) => {
    setDiseaseTags(diseaseTags.filter((tag) => tag !== tagToRemove));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4">
      <div className="absolute top-4 left-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          <ArrowLeft size={20} />
          Back to Main
        </button>
      </div>

      <div className="max-w-4xl mx-auto pt-16">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Upload size={48} className="text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-800">Upload Medical Data</h1>
          </div>
          <p className="text-gray-600">
            Securely upload and share your medical records with end-to-end encryption
          </p>
        </div>

        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 space-y-6">
          {!isWalletConnected && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Wallet size={24} className="text-yellow-600" />
                  <p className="text-yellow-800 font-medium">
                    Wallet not connected. Please connect to upload.
                  </p>
                </div>
                <button
                  onClick={onWalletConnect}
                  className="px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors"
                >
                  Connect Wallet
                </button>
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload File *
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
              <Upload size={48} className="mx-auto mb-4 text-gray-400" />
              <input
                type="file"
                onChange={handleFileChange}
                accept=".xlsx,.xls,.csv,.ods,.tsv,.xlsm,.xlsb,.jpg,.jpeg,.png,.pdf,.dcm,.dicom"
                className="hidden"
                id="file-upload"
                disabled={!isWalletConnected}
              />
              <label
                htmlFor="file-upload"
                className={`cursor-pointer text-blue-600 hover:text-blue-800 font-medium ${
                  !isWalletConnected ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                Click to upload
              </label>
              <p className="text-sm text-gray-500 mt-2">
                Supported: Excel, CSV, Images, PDF, DICOM
              </p>
            </div>
          </div>

          {selectedFile && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <CheckCircle size={20} className="text-green-600" />
                <span className="font-medium text-green-800">
                  {selectedFile.name}
                </span>
                <span className="text-sm text-green-600">
                  ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
            </div>
          )}

          {isGeneratingPreview && (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-sm text-gray-600 mt-2">Generating preview...</p>
            </div>
          )}

          {previewType === 'image' && previewUrl && (
            <div className="border rounded-lg p-4">
              <h3 className="font-medium mb-2">Preview (Anonymized)</h3>
              <img
                src={previewUrl}
                alt="Preview"
                className="max-w-full h-auto rounded"
              />
            </div>
          )}

          {previewType === 'spreadsheet' && previewData && (
            <div className="border rounded-lg p-4 overflow-x-auto">
              <h3 className="font-medium mb-2">
                Spreadsheet Preview ({previewData.previewRows} of{' '}
                {previewData.totalRows} rows)
              </h3>
              <table className="min-w-full border-collapse">
                <thead>
                  <tr>
                    {previewData.headers.map((header, idx) => (
                      <th
                        key={idx}
                        className="border px-4 py-2 bg-gray-100 text-left text-sm font-medium"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.data.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((cell, cellIdx) => (
                        <td key={cellIdx} className="border px-4 py-2 text-sm">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {previewType === 'pdf' && previewUrl && (
            <div className="border rounded-lg p-4">
              <h3 className="font-medium mb-2">PDF Preview</h3>
              <iframe
                src={previewUrl}
                className="w-full h-96 rounded"
                title="PDF Preview"
              />
            </div>
          )}

          {previewType === 'dicom' && previewUrl && (
            <div className="border rounded-lg p-4">
              <h3 className="font-medium mb-2">DICOM Preview</h3>
              <img
                src={previewUrl}
                alt="DICOM Preview"
                className="max-w-full h-auto rounded"
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset Title *
            </label>
            <input
              type="text"
              value={datasetTitle}
              onChange={(e) => setDatasetTitle(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter a descriptive title for your dataset"
              disabled={!isWalletConnected}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description *
            </label>
            <textarea
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              rows={4}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Describe your medical data..."
              disabled={!isWalletConnected}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Disease Tags *
            </label>
            <select
              onChange={handleTagChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={!isWalletConnected}
              value=""
            >
              <option value="">Select disease tags...</option>
              {diseaseOptions.map((disease) => (
                <option key={disease} value={disease}>
                  {disease}
                </option>
              ))}
            </select>
            {diseaseTags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {diseaseTags.map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                  >
                    {tag}
                    <button
                      onClick={() => removeTag(tag)}
                      className="hover:text-blue-900"
                    >
                      <X size={14} />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Data Type *
              </label>
              <select
                value={dataType}
                onChange={(e) => setDataType(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!isWalletConnected}
              >
                <option value="">Select...</option>
                <option value="Personal">Personal</option>
                <option value="Institution">Institution</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Gender
              </label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!isWalletConnected}
              >
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data Source *
            </label>
            <select
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={!isWalletConnected}
            >
              <option value="">Select...</option>
              <option value="Hospital">Hospital</option>
              <option value="Clinic">Clinic</option>
              <option value="Laboratory">Laboratory</option>
              <option value="Imaging Center">Imaging Center</option>
              <option value="Research Institution">Research Institution</option>
              <option value="Personal Records">Personal Records</option>
              <option value="Pharmacy">Pharmacy</option>
              <option value="Telehealth">Telehealth</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Price (ETH) *
            </label>
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              min="0"
              step="0.001"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="0.001"
              disabled={!isWalletConnected}
            />
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-medium text-blue-900 mb-3">Security Features</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2">
                <Shield size={20} className="text-blue-600" />
                <span className="text-sm text-blue-800">End-to-end encryption</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle size={20} className="text-blue-600" />
                <span className="text-sm text-blue-800">Blockchain verification</span>
              </div>
              <div className="flex items-center gap-2">
                <Database size={20} className="text-blue-600" />
                <span className="text-sm text-blue-800">IPFS storage</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleUpload}
            disabled={
              !isWalletConnected ||
              !selectedFile ||
              !datasetTitle ||
              !summary ||
              !diseaseTags.length ||
              !dataType ||
              !dataSource ||
              !price ||
              isUploading
            }
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 font-medium"
          >
            {isUploading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                Uploading...
              </>
            ) : (
              <>
                <Upload size={20} />
                Upload Document
              </>
            )}
          </button>
        </div>
      </div>

      {/* Upload Progress Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">
                {uploadProgress.isComplete
                  ? 'Upload Complete!'
                  : uploadProgress.hasError
                  ? 'Upload Failed'
                  : 'Uploading...'}
              </h2>
              {uploadProgress.isComplete ? (
                <CheckCircle size={32} className="text-green-600" />
              ) : uploadProgress.hasError ? (
                <X size={32} className="text-red-600" />
              ) : (
                <Upload size={32} className="text-blue-600" />
              )}
            </div>

            <div className="space-y-4 mb-6">
              {uploadProgress.steps.map((step, index) => (
                <div key={index} className="flex items-center gap-3">
                  {step.error ? (
                    <X size={20} className="text-red-600 flex-shrink-0" />
                  ) : step.completed ? (
                    <Check size={20} className="text-green-600 flex-shrink-0" />
                  ) : currentStep === index ? (
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 flex-shrink-0"></div>
                  ) : (
                    <Clock size={20} className="text-gray-400 flex-shrink-0" />
                  )}
                  <span
                    className={`text-sm ${
                      step.error
                        ? 'text-red-600'
                        : step.completed
                        ? 'text-green-600'
                        : currentStep === index
                        ? 'text-blue-600 font-medium'
                        : 'text-gray-500'
                    }`}
                  >
                    {step.name}
                  </span>
                </div>
              ))}
            </div>

            {isStreamingEncryption && (
              <div className="mb-6">
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Encryption Progress</span>
                  <span>{encryptionProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${encryptionProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {uploadProgress.isComplete && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 space-y-2 mb-6">
                <div>
                  <span className="text-sm font-medium text-gray-700">
                    IPFS Hash:
                  </span>
                  <p className="text-sm text-gray-600 break-all font-mono">
                    {uploadProgress.ipfsHash}
                  </p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-700">
                    Transaction Hash:
                  </span>
                  <p className="text-sm text-gray-600 break-all font-mono">
                    {uploadProgress.transactionHash}
                  </p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-700">Price:</span>
                  <p className="text-sm text-gray-600">{uploadProgress.price} ETH</p>
                </div>
              </div>
            )}

            {uploadProgress.hasError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <p className="text-sm text-red-600">{uploadProgress.errorMessage}</p>
              </div>
            )}

            <div className="flex gap-4">
              {uploadProgress.isComplete && (
                <button
                  onClick={closeModal}
                  className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Upload Another
                </button>
              )}
              {uploadProgress.hasError && (
                <button
                  onClick={() => {
                    closeModal();
                    handleUpload();
                  }}
                  className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Try Again
                </button>
              )}
              {!uploadProgress.isComplete && !uploadProgress.hasError && (
                <button
                  onClick={closeModal}
                  className="flex-1 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
