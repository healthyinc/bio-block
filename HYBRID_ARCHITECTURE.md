# Hybrid Architecture Implementation

## Overview

The Bio-Block system now uses a hybrid approach for file uploads:

- **Frontend**: Handles encryption (client-side security)
- **Backend**: Handles IPFS uploads (API key security & reliability)

## Architecture Flow

```
User File → Frontend Encryption → Backend IPFS Upload → Blockchain Storage
     ↓              ↓                      ↓                    ↓
  Original      Encrypted Data       IPFS Hash            Transaction Hash
```

## Implementation Details

### Frontend Changes (`upload_data.js`)

1. **Removed direct IPFS upload**: No longer uses `uploadToIPFS` from `UploadFile.js`
2. **Added hybrid function**: `uploadToIPFSViaBackend()` sends encrypted data to backend
3. **Client-side encryption**: Still encrypts files locally using `encryptFile()`
4. **Updated preview upload**: Preview files also use backend route

### Backend Changes (`javascript_backend`)

1. **New controller**: `controllers/ipfsController.js` handles IPFS uploads
2. **New route**: `/api/ipfs/upload` endpoint for encrypted file uploads
3. **Secure API keys**: Pinata JWT stored in backend environment variables
4. **Added dependency**: `form-data` package for multipart uploads to Pinata

### Security Benefits

- ✅ **Client-side encryption**: Files encrypted before leaving user's browser
- ✅ **API key security**: Pinata credentials hidden in backend
- ✅ **No plaintext on server**: Backend only handles encrypted blobs
- ✅ **Reliable uploads**: Server-side upload stability

### Configuration Required

#### Backend Environment Variables

```env
PINATA_JWT=your_pinata_jwt_token_here
PORT=3001
```

#### Frontend Environment Variables (unchanged)

```env
REACT_APP_JS_BACKEND_URL=http://localhost:3001
REACT_APP_ENCRYPTION_KEY=your_32_byte_encryption_key
```

## API Endpoints

### New IPFS Upload Endpoint

```
POST /api/ipfs/upload
Content-Type: multipart/form-data

Body:
- encryptedFile: Blob (encrypted file data)
- fileName: String (original filename)

Response:
{
  "success": true,
  "ipfsHash": "QmXXXXX...",
  "fileName": "document.xlsx",
  "fileSize": 1024000
}
```

## Benefits of Hybrid Approach

1. **Security**: Best of both worlds - client encryption + server reliability
2. **Scalability**: Backend can handle large files better than browser
3. **API Key Protection**: Sensitive credentials never exposed to frontend
4. **Reliability**: Server-grade network for IPFS uploads
5. **Consistency**: Predictable upload behavior across different clients

## Migration Notes

- Existing encrypted files in IPFS remain compatible
- Frontend encryption logic unchanged
- Only upload mechanism modified
- All existing features (search, download, preview) work unchanged

## Testing

To test the hybrid implementation:

1. Start JavaScript backend: `npm start` (in javascript_backend folder)
2. Start React frontend: `npm start` (in prototype folder)
3. Upload a file through the UI
4. Verify encryption happens in browser (client-side)
5. Verify IPFS upload happens via backend (check backend logs)
6. Confirm file can be downloaded and decrypted successfully

The hybrid approach maintains all security benefits while improving reliability and protecting API credentials.
