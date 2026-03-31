import CryptoJS from 'crypto-js';

class StreamingEncryption {
  constructor(chunkSize = 1024 * 1024) {
    this.chunkSize = chunkSize;
    // All encryption now handled server-side
  }

  getOptimalChunkSize(fileSize) {
    if (fileSize < 10 * 1024 * 1024) return 512 * 1024;
    if (fileSize < 100 * 1024 * 1024) return 1024 * 1024;
    if (fileSize < 1024 * 1024 * 1024) return 2 * 1024 * 1024;
    return 4 * 1024 * 1024;
  }

  async encryptFileStream(file, progressCallback = null) {
    const optimalChunkSize = this.getOptimalChunkSize(file.size);
    const totalChunks = Math.ceil(file.size / optimalChunkSize);
    
    console.log(`Starting server-side streaming encryption: ${file.size} bytes, ${totalChunks} chunks`);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunkSize', optimalChunkSize.toString());
    
    const response = await fetch('/api/encrypt-stream', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      },
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Server-side encryption failed: ' + response.statusText);
    }
    
    const reader = response.body.getReader();
    const contentLength = +response.headers.get('Content-Length');
    let receivedLength = 0;
    let chunks = [];
    
    while(true) {
      const {done, value} = await reader.read();
      
      if (done) break;
      
      chunks.push(value);
      receivedLength += value.length;
      
      if (progressCallback) {
        const progress = (receivedLength / contentLength) * 100;
        progressCallback(progress);
      }
      
      await this.sleep(5);
    }
    
    const result = new Uint8Array(receivedLength);
    let position = 0;
    for(let chunk of chunks) {
      result.set(chunk, position);
      position += chunk.length;
    }
    
    console.log('Server-side streaming encryption completed');
    return result;
  }

  async decryptFileStream(encryptedData, progressCallback = null) {
    try {
      const formData = new FormData();
      formData.append('encryptedFile', new Blob([encryptedData], { type: 'application/octet-stream' }));
      
      const response = await fetch('/api/decrypt-stream', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Server-side decryption failed: ' + response.statusText);
      }
      
      const reader = response.body.getReader();
      const contentLength = +response.headers.get('Content-Length');
      let receivedLength = 0;
      let chunks = [];
      
      while(true) {
        const {done, value} = await reader.read();
        
        if (done) break;
        
        chunks.push(value);
        receivedLength += value.length;
        
        if (progressCallback) {
          const progress = (receivedLength / contentLength) * 100;
          progressCallback(progress);
        }
        
        await this.sleep(5);
      }
      
      const result = new Uint8Array(receivedLength);
      let position = 0;
      for(let chunk of chunks) {
        result.set(chunk, position);
        position += chunk.length;
      }
      
      console.log('Server-side decryption completed');
      return result;
      
    } catch (error) {
      console.error('Stream decryption error:', error);
      throw new Error(`Stream decryption failed: ${error.message}`);
    }
  }

  // Convert CryptoJS WordArray to Uint8Array
  wordArrayToUint8Array(wordArray) {
    const words = wordArray.words;
    const sigBytes = wordArray.sigBytes;
    const u8 = new Uint8Array(sigBytes);
    
    for (let i = 0; i < sigBytes; i++) {
      u8[i] = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
    }
    
    return u8;
  }

  // Efficiently combine chunks into single Uint8Array
  combineChunks(chunks) {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    
    return result;
  }

  // Non-blocking delay to keep UI responsive
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Check if file should use streaming (based on size)
  shouldUseStreaming(fileSize) {
    const STREAMING_THRESHOLD = 5 * 1024 * 1024; // 5MB
    const shouldStream = fileSize > STREAMING_THRESHOLD;
    console.log(`shouldUseStreaming: fileSize=${fileSize} bytes (${(fileSize / (1024*1024)).toFixed(2)}MB), threshold=${STREAMING_THRESHOLD} bytes (${(STREAMING_THRESHOLD / (1024*1024)).toFixed(2)}MB), shouldStream=${shouldStream}`);
    return shouldStream;
  }

  // Get memory-safe chunk count estimation
  getChunkCount(fileSize) {
    const chunkSize = this.getOptimalChunkSize(fileSize);
    return Math.ceil(fileSize / chunkSize);
  }
}

export default StreamingEncryption;
