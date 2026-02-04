import React, { useState, useEffect } from 'react';
import { Upload, Wallet, ArrowLeft, Shield, Database, CheckCircle, X, Clock, Check } from 'lucide-react';
import { storeDocumentHash } from './contractService';
import { encryptFile } from './encryptionUtils.js';
import StreamingEncryption from './utils/streamingEncryption.js';

export default function UploadData({ onBack, isWalletConnected, walletAddress, onWalletConnect }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [datasetTitle, setDatasetTitle] = useState('');
  const [summary, setSummary] = useState('');
  const [diseaseTags, setDiseaseTags] = useState([]);
  const [dataType, setDataType] = useState('');
  const [gender, setGender] = useState('');
  const [dataSource, setDataSource] = useState('');
  const [price, setPrice] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewType, setPreviewType] = useState(null); // 'image', 'spreadsheet', 'pdf', 'dicom'
  const [previewData, setPreviewData] = useState(null); // For spreadsheet data
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);

  // Cleanup preview URL when it changes or component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Cleanup preview data when component unmounts
  useEffect(() => {
    return () => {
      setPreviewUrl(null);
      setPreviewType(null);
      setPreviewData(null);
    };
  }, []);

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
    'Alzheimer\'s Disease',
    'Parkinson\'s Disease',
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
    'Other'
  ];
  
  // Modal and progress states
  const [showModal, setShowModal] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [encryptionProgress, setEncryptionProgress] = useState(0);
  const [isStreamingEncryption, setIsStreamingEncryption] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({
    steps: [
      { name: 'Preparing file...', completed: false, error: false },
      { name: 'Anonymizing data (if Excel/Image)...', completed: false, error: false },
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
    errorMessage: ''
  });

  // Find your existing file change handler and REPLACE it with this:
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    
    // --- 1. Memory Leak Prevention ---
    // If a preview URL already exists, revoke it before creating a new one
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    
    // Reset preview state
    setPreviewType(null);
    setPreviewData(null);
    
    // Set the selected file for the upload form
    setSelectedFile(file);

    if (!file) {
      return; // No file selected
    }

    const jsBackendUrl = process.env.REACT_APP_JS_BACKEND_URL || 'http://localhost:3001';
    setIsGeneratingPreview(true);

    try {
      // --- 2. Check File Type for Preview ---
      const fileExtension = file.name.split('.').pop()?.toLowerCase();
      const isSpreadsheet = file.type.includes('spreadsheet') || 
                          file.type.includes('csv') || 
                          ['xlsx', 'xls', 'csv', 'ods', 'tsv', 'xlsm', 'xlsb'].includes(fileExtension);
      const isPdf = file.type === 'application/pdf' || fileExtension === 'pdf';
      const isDicom = fileExtension === 'dcm' || fileExtension === 'dicom';

      if (file.type.startsWith('image/')) {
        // Image preview
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/image`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Preview generation failed');
        }

        const imageBlob = await response.blob();
        const objectUrl = URL.createObjectURL(imageBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('image');

      } else if (isSpreadsheet) {
        // Spreadsheet preview
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/spreadsheet`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Preview generation failed');
        }

        const spreadsheetData = await response.json();
        setPreviewData(spreadsheetData);
        setPreviewType('spreadsheet');

      } else if (isPdf) {
        // PDF preview
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/pdf`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Preview generation failed');
        }

        const pdfBlob = await response.blob();
        const objectUrl = URL.createObjectURL(pdfBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('pdf');

      } else if (isDicom) {
        // DICOM preview
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${jsBackendUrl}/api/preview/dicom`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Preview generation failed');
        }

        const imageBlob = await response.blob();
        const objectUrl = URL.createObjectURL(imageBlob);
        setPreviewUrl(objectUrl);
        setPreviewType('dicom');

      } else {
        // Unsupported file type
        console.log("File type not supported for preview:", file.type);
      }

    } catch (error) {
      console.error('Error generating preview:', error);
      // Show error to user
      alert(`Preview generation failed: ${error.message}. Please check console for details.`);
      setPreviewUrl(null);
      setPreviewType(null);
      setPreviewData(null);
    } finally {
      setIsGeneratingPreview(false);
    }
  };

  const handleSummaryChange = (event) => {
    setSummary(event.target.value);
  };

  const handlePriceChange = (event) => {
    setPrice(event.target.value);
  };

  const resetProgress = () => {
    setUploadProgress({
      steps: [
        { name: 'Preparing file...', completed: false, error: false },
        { name: 'Anonymizing data (if Excel/Image)...', completed: false, error: false },
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
      errorMessage: ''
    });
    setCurrentStep(0);
  };

  const updateStep = (stepIndex, completed = false, error = false) => {
    setUploadProgress(prev => ({
      ...prev,
      steps: prev.steps.map((step, index) => 
        index === stepIndex ? { ...step, completed, error } : step
      )
    }));
    
    if (!completed && !error) {
      // This step is now active/in progress
      setCurrentStep(stepIndex);
    } else if (completed && !error) {
      // This step is completed, move to next step
      setCurrentStep(stepIndex + 1);
    }
  };

  const setError = (message) => {
    setUploadProgress(prev => ({
      ...prev,
      hasError: true,
      errorMessage: message
    }));
  };

  const closeModal = () => {
    setShowModal(false);
    if (uploadProgress.isComplete) {
      // Reset all fields if upload was successful
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

  const anonymizeFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    if (dataType === 'Personal') {
      formData.append('walletAddress', walletAddress);
    }

    // Request preview generation for Excel files
    const shouldGeneratePreview = file.name.match(/\.(xlsx|xls)$/i);
    if (shouldGeneratePreview) {
      formData.append('generatePreview', 'true');
    }

    let backendUrl, endpoint;
    
    if (file.type.startsWith('image/')) {
      backendUrl = process.env.REACT_APP_PYTHON_BACKEND_URL || 'http://localhost:3002';
      endpoint = '/anonymize_image';
    } else if (file.name.match(/\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i)) {
      backendUrl = process.env.REACT_APP_JS_BACKEND_URL || 'http://localhost:3001';
      endpoint = '/api/anonymize';
    } else {
      throw new Error('File type not supported for anonymization. Only Excel (.xlsx, .xls, .csv, .ods, .tsv, .xlsm, .xlsb) and image files (.jpg, .jpeg, .png) are supported.');
    }

    const response = await fetch(`${backendUrl}${endpoint}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || 'File anonymization failed');
    }

    // Handle different response types
    if (shouldGeneratePreview && response.headers.get('content-type')?.includes('application/json')) {
      // Response contains both main file and preview
      const jsonResponse = await response.json();
      
      // Convert base64 back to File objects
      const mainFileBuffer = Uint8Array.from(atob(jsonResponse.files.main.buffer), c => c.charCodeAt(0));
      const mainFile = new File([mainFileBuffer], jsonResponse.files.main.filename, { 
        type: jsonResponse.files.main.contentType 
      });

      const previewFileBuffer = Uint8Array.from(atob(jsonResponse.files.preview.buffer), c => c.charCodeAt(0));
      const previewFile = new File([previewFileBuffer], jsonResponse.files.preview.filename, { 
        type: jsonResponse.files.preview.contentType 
      });

      return { mainFile, previewFile };
    } else {
      // Standard blob response for images or Excel without preview
      const blob = await response.blob();
      const mainFile = new File([blob], `anonymized_${file.name}`, { type: file.type });
      return { mainFile, previewFile: null };
    }
  };

  // New hybrid approach: Upload encrypted file to backend for IPFS upload
  const uploadToIPFSViaBackend = async (encryptedData, fileName) => {
    const formData = new FormData();
    formData.append('encryptedFile', new Blob([encryptedData], { type: 'application/octet-stream' }));
    formData.append('fileName', fileName);

    const backendUrl = process.env.REACT_APP_JS_BACKEND_URL || 'http://localhost:3001';
    
    const response = await fetch(`${backendUrl}/api/ipfs/upload`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || 'IPFS upload failed');
    }

    const result = await response.json();
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
      updateStep(0); // Mark as in progress
      await new Promise(resolve => setTimeout(resolve, 500)); // Small delay to show progress
      let fileToUpload = selectedFile;
      let previewFile = null;
      updateStep(0, true); // Mark as completed
      
      // Step 2: Anonymizing (if Excel or Image) and extracting preview
      updateStep(1); // Mark as in progress
      const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();
      const isSpreadsheet = selectedFile.name.match(/\.(xlsx|xls|csv|ods|tsv|xlsm|xlsb)$/i);
      const isImage = selectedFile.type.startsWith('image/');
      const isPdf = selectedFile.type === 'application/pdf' || fileExtension === 'pdf';
      const isDicom = fileExtension === 'dcm' || fileExtension === 'dicom';
      
      if (isSpreadsheet || isImage) {
        // Anonymize spreadsheets and images
        try {
          const result = await anonymizeFile(selectedFile);
          fileToUpload = result.mainFile;
          previewFile = result.previewFile; // Will be null for images, populated for Excel
          updateStep(1, true, false); // Mark as completed
        } catch (error) {
          updateStep(1, false, true); // Mark as error
          setError(`Anonymization failed: ${error.message}`);
          return;
        }
      } else if (isPdf || isDicom) {
        // PDF and DICOM files: Skip anonymization for now (can be added in future)
        // Use file as-is for upload
        fileToUpload = selectedFile;
        previewFile = null;
        updateStep(1, true, false); // Mark as completed (skipped anonymization)
      } else {
        // File type not supported - this should not happen due to file input restrictions
        updateStep(1, false, true); // Mark as error
        setError('File type not supported for upload. Please select a supported file type.');
        return;
      }
      
      // Step 3: Encrypting file (with streaming for large files)
      updateStep(2); // Mark as in progress
      try {
        const streamer = new StreamingEncryption();
        
        console.log(`=== ENCRYPTION DECISION ===`);
        console.log(`Original file: ${selectedFile.name}, size: ${selectedFile.size} bytes (${(selectedFile.size / (1024*1024)).toFixed(2)}MB)`);
        console.log(`Anonymized file: ${fileToUpload.name}, size: ${fileToUpload.size} bytes (${(fileToUpload.size / (1024*1024)).toFixed(2)}MB)`);
        
        const shouldUseStreaming = streamer.shouldUseStreaming(fileToUpload.size);
        
        console.log(`Final decision: streaming = ${shouldUseStreaming}`);
        console.log(`=== END ENCRYPTION DECISION ===`);
        
        let encryptedFile;
        
        if (shouldUseStreaming) {
          // Use streaming encryption for large files
          console.log(`🔄 Using STREAMING encryption for large file (${(fileToUpload.size / (1024*1024)).toFixed(2)}MB)`);
          setIsStreamingEncryption(true);
          setEncryptionProgress(0);
          
          encryptedFile = await streamer.encryptFileStream(
            fileToUpload,
            (progress) => {
              setEncryptionProgress(progress);
              console.log(`Encryption progress: ${progress.toFixed(1)}%`);
            }
          );
          
          setIsStreamingEncryption(false);
          console.log(`✅ Streaming encryption completed`);
        } else {
          // Use traditional encryption for small files
          console.log(`⚡ Using TRADITIONAL encryption for small file (${(fileToUpload.size / (1024*1024)).toFixed(2)}MB)`);
          const fileBuffer = await fileToUpload.arrayBuffer();
          encryptedFile = encryptFile(new Uint8Array(fileBuffer));
          console.log(`✅ Traditional encryption completed`);
        }
        
        updateStep(2, true, false); // Mark as completed
        
        // Step 4: Uploading to IPFS via backend
        updateStep(3); // Mark as in progress
        const result = await uploadToIPFSViaBackend(encryptedFile, fileToUpload.name);
        if (!result.success) {
          updateStep(3, false, true); // Mark as error
          setError(`IPFS upload failed: ${result.error}`);
          return;
        }
        updateStep(3, true, false); // Mark as completed
        
        // Update progress with IPFS hash
        setUploadProgress(prev => ({
          ...prev,
          ipfsHash: result.ipfsHash
        }));
        
        updateStep(4); 
        const txHash = await storeDocumentHash(result.ipfsHash, price);
        updateStep(4, true, false); 
        
        setUploadProgress(prev => ({
          ...prev,
          transactionHash: txHash,
          price: price
        }));
        
        // Step 6: Saving metadata
        updateStep(5); // Mark as in progress
        
        let previewHash = null;
        // Upload preview to IPFS via backend if it exists
        if (previewFile) {
          try {
            const previewStreamer = new StreamingEncryption();
            const shouldUseStreamingForPreview = previewStreamer.shouldUseStreaming(previewFile.size);
            
            let encryptedPreview;
            
            if (shouldUseStreamingForPreview) {
              // Use streaming for large preview files
              encryptedPreview = await previewStreamer.encryptFileStream(
                previewFile,
                (progress) => console.log(`Preview encryption progress: ${progress.toFixed(1)}%`)
              );
            } else {
              // Use traditional encryption for small preview files
              const previewBuffer = await previewFile.arrayBuffer();
              encryptedPreview = encryptFile(new Uint8Array(previewBuffer));
            }
            
            const previewResult = await uploadToIPFSViaBackend(encryptedPreview, previewFile.name);
            if (previewResult.success) {
              previewHash = previewResult.ipfsHash;
            }
          } catch (error) {
            console.warn('Preview upload failed, continuing without preview:', error);
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
          ...(previewHash && { previewHash: previewHash }),
          ...(diseaseTags.length && { disease_tags: diseaseTags.join(', ') })
        };

        const pythonBackendUrl = process.env.REACT_APP_PYTHON_BACKEND_URL || 'http://localhost:3002';
        const storeResponse = await fetch(`${pythonBackendUrl}/store`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            summary: summary.trim(),
            dataset_title: datasetTitle.trim(),
            cid: result.ipfsHash,
            metadata: metadata
          }),
        });

        if (!storeResponse.ok) {
          updateStep(5, false, true); // Mark as error
          setError(`Metadata storage failed: ${storeResponse.statusText}`);
          return;
        }

        await storeResponse.json(); // Process response but don't need to store
        updateStep(5, true, false); // Mark as completed
        
        // Mark as complete
        setUploadProgress(prev => ({
          ...prev,
          isComplete: true
        }));
        
      } catch (error) {
        const currentStepIndex = currentStep;
        updateStep(currentStepIndex, false, true); // Mark current step as error
        setError(error.message);
      }
      
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Enhanced Header with Gradient */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-blue-600 to-indigo-600 h-2"></div>
      
      {/* Back Button */}
      <div className="absolute top-6 left-6">
        <button
          onClick={onBack}
          className="flex items-center gap-3 px-5 py-3 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-all duration-200 shadow-lg hover:shadow-xl backdrop-blur-sm"
        >
          <ArrowLeft size={18} className="text-gray-600" />
          <span className="font-medium text-gray-700">Back to Main</span>
        </button>
      </div>

      <div className="flex items-center justify-center min-h-screen p-6">
        <div className="w-full max-w-2xl">
          
          {/* Header Section */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl mb-4 shadow-lg">
              <Upload size={28} className="text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-2">
              Upload Document
            </h1>
            <p className="text-gray-600 max-w-lg mx-auto">
              Securely upload your medical documents to the blockchain with end-to-end encryption
            </p>
          </div>

          {/* Main Upload Card */}
          <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl border border-gray-100">
            
            {!isWalletConnected && (
              <div className="p-8 border-b border-gray-100">
                <div className="flex items-center gap-4 p-6 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-2xl">
                  <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center flex-shrink-0">
                    <Wallet size={24} className="text-amber-600" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-amber-800 mb-1">Wallet Connection Required</h3>
                    <p className="text-amber-700 text-sm mb-3">
                      Connect your wallet to upload and manage documents securely
                    </p>
                    <button
                      onClick={onWalletConnect}
                      className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-md hover:shadow-lg text-sm font-medium"
                    >
                      Connect Wallet
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="p-8">
              {/* File Upload Section */}
              <div className="mb-8">
                <label className="block text-lg font-semibold text-gray-800 mb-4">
                  📄 Select Document
                </label>
                <div className={`relative border-2 border-dashed ${isWalletConnected ? 'border-gray-300 hover:border-blue-400 hover:bg-blue-50/50' : 'border-gray-200 bg-gray-50'} rounded-2xl p-8 text-center transition-all duration-200`}>
                  <div className={`${isWalletConnected ? 'text-gray-500' : 'text-gray-300'} mb-4`}>
                    <Upload className="mx-auto mb-4" size={48} />
                  </div>
                  
                  <input
                    type="file"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    id="file-upload"
                    accept=".xlsx,.xls,.csv,.ods,.tsv,.xlsm,.xlsb,.jpg,.jpeg,.png,.pdf,.dcm,.dicom"
                    disabled={!isWalletConnected}
                  />
                  
                  <div className="pointer-events-none">
                    <p className={`${isWalletConnected ? 'text-blue-600' : 'text-gray-400'} font-medium mb-2`}>
                      {selectedFile ? selectedFile.name : 'Choose a file or drag and drop'}
                    </p>
                    <p className={`${isWalletConnected ? 'text-gray-500' : 'text-gray-400'} text-sm`}>
                      Spreadsheets, Images (JPG, JPEG, PNG), PDFs, and DICOM files
                    </p>
                  </div>
                </div>

                {selectedFile && (
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl">
                    <div className="flex items-center gap-3">
                      <CheckCircle size={20} className="text-green-600 flex-shrink-0" />
                      <div>
                        <p className="font-medium text-green-800">
                          {selectedFile.name}
                        </p>
                        <p className="text-sm text-green-600">
                          Size: {(selectedFile.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* --- START: PREVIEW BLOCK --- */}
          <div className="mb-4">
            {isGeneratingPreview && (
              <div className="text-center p-4">
                <p className="text-gray-400">Generating preview...</p>
              </div>
            )}
            
            {previewType === 'image' && previewUrl && !isGeneratingPreview && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
                <h4 className="text-lg font-semibold text-gray-200 mb-2">
                  Preview:
                </h4>
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="w-full max-w-md rounded-lg border-2 border-gray-600"
                />
              </div>
            )}

            {previewType === 'spreadsheet' && previewData && !isGeneratingPreview && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
                <h4 className="text-lg font-semibold text-gray-200 mb-2">
                  Preview: {previewData.fileName}
                </h4>
                {previewData.message && (
                  <p className="text-sm text-gray-400 mb-3">{previewData.message}</p>
                )}
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white text-gray-800 rounded-lg border border-gray-300">
                    <thead className="bg-gray-100">
                      <tr>
                        {previewData.headers && previewData.headers.map((header, idx) => (
                          <th key={idx} className="px-4 py-2 text-left border-b border-gray-300 font-semibold">
                            {header || `Column ${idx + 1}`}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.data && previewData.data.map((row, rowIdx) => (
                        <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          {previewData.headers && previewData.headers.map((_, colIdx) => (
                            <td key={colIdx} className="px-4 py-2 border-b border-gray-200">
                              {row[colIdx] !== undefined && row[colIdx] !== null ? String(row[colIdx]) : ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {previewData.totalRows > previewData.previewRows && (
                  <p className="text-xs text-gray-400 mt-2">
                    Showing {previewData.previewRows} of {previewData.totalRows} rows
                  </p>
                )}
              </div>
            )}

            {previewType === 'pdf' && previewUrl && !isGeneratingPreview && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
                <h4 className="text-lg font-semibold text-gray-200 mb-2">
                  Preview:
                </h4>
                <div className="w-full border-2 border-gray-600 rounded-lg overflow-hidden" style={{ maxHeight: '600px' }}>
                  <iframe
                    src={previewUrl}
                    title="PDF Preview"
                    className="w-full h-full"
                    style={{ minHeight: '500px', border: 'none' }}
                  />
                </div>
              </div>
            )}

            {previewType === 'dicom' && previewUrl && !isGeneratingPreview && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
                <h4 className="text-lg font-semibold text-gray-200 mb-2">
                  Preview:
                </h4>
                <img 
                  src={previewUrl} 
                  alt="DICOM Preview" 
                  className="w-full max-w-md rounded-lg border-2 border-gray-600"
                />
              </div>
            )}
          </div>
          {/* --- END: PREVIEW BLOCK --- */}

              {/* Form Fields Grid */}
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                
                {/* Dataset Title */}
                <div className="md:col-span-2">
                  <label htmlFor="datasetTitle" className="block text-sm font-semibold text-gray-800 mb-2">
                    📋 Dataset Title *
                  </label>
                  <input
                    id="datasetTitle"
                    type="text"
                    value={datasetTitle}
                    onChange={(e) => setDatasetTitle(e.target.value)}
                    placeholder="Enter a unique title for your dataset..."
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    disabled={!isWalletConnected}
                  />
                </div>

                {/* Description */}
                <div className="md:col-span-2">
                  <label htmlFor="summary" className="block text-sm font-semibold text-gray-800 mb-2">
                    📝 Description *
                  </label>
                  <textarea
                    id="summary"
                    value={summary}
                    onChange={handleSummaryChange}
                    placeholder="Provide a detailed description of what this document contains..."
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    rows={4}
                    disabled={!isWalletConnected}
                  />
                </div>

                {/* Disease Tags */}
                <div className="md:col-span-2">
                  <label htmlFor="diseaseTags" className="block text-sm font-semibold text-gray-800 mb-2">
                    🏷️ Disease Tags *
                  </label>
                  <div className="relative">
                    <select
                      id="diseaseTags"
                      onChange={(e) => {
                        const selectedValue = e.target.value;
                        if (selectedValue && !diseaseTags.includes(selectedValue)) {
                          setDiseaseTags([...diseaseTags, selectedValue]);
                        }
                        e.target.value = ''; // Reset dropdown
                      }}
                      className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                      disabled={!isWalletConnected}
                    >
                      <option value="">Select disease tags to add...</option>
                      {diseaseOptions.map((disease) => (
                        <option key={disease} value={disease} disabled={diseaseTags.includes(disease)}>
                          {disease}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Selected Tags Display */}
                  {diseaseTags.length > 0 && (
                    <div className="mt-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
                      <p className="text-sm font-medium text-blue-800 mb-2">Selected Disease Tags:</p>
                      <div className="flex flex-wrap gap-2">
                        {diseaseTags.map((tag, index) => (
                          <span
                            key={index}
                            className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                          >
                            {tag}
                            <button
                              type="button"
                              onClick={() => setDiseaseTags(diseaseTags.filter((_, i) => i !== index))}
                              className="ml-1 hover:bg-blue-200 rounded-full p-1 transition-colors duration-200"
                              title="Remove tag"
                            >
                              <X size={12} />
                            </button>
                          </span>
                        ))}
                      </div>
                      <button
                        type="button"
                        onClick={() => setDiseaseTags([])}
                        className="mt-2 text-xs text-blue-600 hover:text-blue-700 font-medium"
                      >
                        Clear all tags
                      </button>
                    </div>
                  )}
                  
                  <p className="text-xs text-gray-500 mt-2">
                    Select multiple disease tags that apply to your document. Click on a tag to remove it.
                  </p>
                </div>

                {/* Data Type */}
                <div>
                  <label htmlFor="dataType" className="block text-sm font-semibold text-gray-800 mb-2">
                    🔐 Data Type *
                  </label>
                  <select
                    id="dataType"
                    value={dataType}
                    onChange={(e) => setDataType(e.target.value)}
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    disabled={!isWalletConnected}
                  >
                    <option value="">Select data type</option>
                    <option value="Personal">Personal</option>
                    <option value="Institution">Institution</option>
                  </select>
                </div>

                {/* Gender */}
                <div>
                  <label htmlFor="gender" className="block text-sm font-semibold text-gray-800 mb-2">
                    👤 Gender
                  </label>
                  <select
                    id="gender"
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    disabled={!isWalletConnected}
                  >
                    <option value="">Select gender (optional)</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                {/* Data Source */}
                <div>
                  <label htmlFor="dataSource" className="block text-sm font-semibold text-gray-800 mb-2">
                    🏥 Data Source *
                  </label>
                  <select
                    id="dataSource"
                    value={dataSource}
                    onChange={(e) => setDataSource(e.target.value)}
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    disabled={!isWalletConnected}
                  >
                    <option value="">Select data source</option>
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

                {/* Price */}
                <div>
                  <label htmlFor="price" className="block text-sm font-semibold text-gray-800 mb-2">
                    💰 Price (ETH) *
                  </label>
                  <input
                    id="price"
                    type="number"
                    step="0.001"
                    min="0"
                    value={price}
                    onChange={handlePriceChange}
                    placeholder="0.01"
                    className={`w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 ${!isWalletConnected ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}
                    disabled={!isWalletConnected}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Set the price users will pay to download your document
                  </p>
                </div>
              </div>

              {/* Security Features */}
              <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl border border-blue-100">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Shield size={20} className="text-blue-600" />
                  Security Features
                </h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-gray-700">End-to-end encryption</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-gray-700">Blockchain verification</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-gray-700">IPFS decentralized storage</span>
                  </div>
                </div>
              </div>

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={!selectedFile || !isWalletConnected || !datasetTitle.trim() || !summary.trim() || !diseaseTags.length || !dataType || !dataSource || !price || parseFloat(price) <= 0 || isUploading}
                className="w-full px-6 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-2xl hover:from-green-700 hover:to-emerald-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none font-semibold text-lg"
              >
                {isUploading ? (
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Uploading to Blockchain...
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-3">
                    <Database size={20} />
                    Upload to IPFS & Blockchain
                  </div>
                )}
              </button>
            </div>
          </div>

          {/* Footer Note */}
          <div className="mt-8 text-center">
            <p className="text-sm text-gray-500">
              Your document will be encrypted and stored securely on IPFS with blockchain verification
            </p>
          </div>

        </div>
      </div>

      {/* Upload Progress Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 relative">
            {/* Modal Header */}
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl mb-4 shadow-lg">
                {uploadProgress.isComplete ? (
                  <CheckCircle size={32} className="text-white" />
                ) : uploadProgress.hasError ? (
                  <X size={32} className="text-white" />
                ) : (
                  <Upload size={32} className="text-white" />
                )}
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-2">
                {uploadProgress.isComplete 
                  ? 'Upload Complete!' 
                  : uploadProgress.hasError 
                    ? 'Upload Failed' 
                    : 'Uploading Document'}
              </h3>
              {!uploadProgress.hasError && !uploadProgress.isComplete && (
                <p className="text-gray-600">Please wait while we process your document...</p>
              )}
            </div>

            {/* Progress Steps */}
            <div className="space-y-4 mb-6">
              {uploadProgress.steps.map((step, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                    step.error 
                      ? 'bg-red-500' 
                      : step.completed 
                        ? 'bg-green-500' 
                        : index === currentStep
                          ? 'bg-blue-500'
                          : 'bg-gray-300'
                  }`}>
                    {step.error ? (
                      <X size={14} className="text-white" />
                    ) : step.completed ? (
                      <Check size={14} className="text-white" />
                    ) : index === currentStep ? (
                      <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      <Clock size={14} className="text-white" />
                    )}
                  </div>
                  <span className={`text-sm ${
                    step.error 
                      ? 'text-red-600 font-medium' 
                      : step.completed 
                        ? 'text-green-600 font-medium' 
                        : index === currentStep
                          ? 'text-blue-600 font-medium'
                          : 'text-gray-500'
                  }`}>
                    {step.name}
                    {/* Show streaming encryption progress */}
                    {index === 2 && isStreamingEncryption && encryptionProgress > 0 && (
                      <div className="mt-2">
                        <div className="flex justify-between text-xs text-blue-600 mb-1">
                          <span>Streaming encryption...</span>
                          <span>{encryptionProgress.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-blue-100 rounded-full h-1.5">
                          <div 
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300" 
                            style={{ width: `${encryptionProgress}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </span>
                </div>
              ))}
            </div>

            {/* Success Information */}
            {uploadProgress.isComplete && (
              <div className="space-y-3 mb-6 p-4 bg-green-50 rounded-2xl border border-green-200">
                <div className="text-sm">
                  <span className="font-medium text-green-800">IPFS Hash:</span>
                  <p className="text-green-600 break-all font-mono text-xs mt-1">{uploadProgress.ipfsHash}</p>
                </div>
                <div className="text-sm">
                  <span className="font-medium text-green-800">Transaction:</span>
                  <p className="text-green-600 break-all font-mono text-xs mt-1">{uploadProgress.transactionHash}</p>
                </div>
                <div className="text-sm">
                  <span className="font-medium text-green-800">Price:</span>
                  <span className="text-green-600 font-semibold ml-2">{uploadProgress.price} ETH</span>
                </div>
              </div>
            )}

            {/* Error Information */}
            {uploadProgress.hasError && (
              <div className="mb-6 p-4 bg-red-50 rounded-2xl border border-red-200">
                <p className="text-red-800 font-medium mb-2">Error occurred:</p>
                <p className="text-red-600 text-sm">{uploadProgress.errorMessage}</p>
              </div>
            )}

            {/* Modal Actions */}
            <div className="flex gap-3">
              {uploadProgress.isComplete || uploadProgress.hasError ? (
                <button
                  onClick={closeModal}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-lg hover:shadow-xl font-medium"
                >
                  {uploadProgress.isComplete ? 'Upload Another Document' : 'Try Again'}
                </button>
              ) : (
                <div className="flex-1 px-6 py-3 bg-gray-100 text-gray-400 rounded-2xl font-medium text-center cursor-not-allowed">
                  Upload in Progress...
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}