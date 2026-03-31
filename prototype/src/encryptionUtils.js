// encryptionUtils.js - Server-side encryption only
// All encryption now handled server-side to prevent key exposure

export const encryptFile = async (fileBuffer) => {
  // Send file to backend for secure encryption
  const formData = new FormData();
  formData.append('file', new Blob([fileBuffer], { type: 'application/octet-stream' }));
  
  const response = await fetch('/api/encrypt', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    },
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Encryption failed: ' + response.statusText);
  }
  
  const result = await response.json();
  return new Blob([result.encryptedData], { type: 'application/octet-stream' });
};

export const decryptFile = async (encryptedData) => {
  // Send encrypted data to backend for secure decryption
  const formData = new FormData();
  formData.append('encryptedFile', new Blob([encryptedData], { type: 'application/octet-stream' }));
  
  const response = await fetch('/api/decrypt', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('authToken')}`
    },
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Decryption failed: ' + response.statusText);
  }
  
  const result = await response.json();
  return result.decryptedData;
};