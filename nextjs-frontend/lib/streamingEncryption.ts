import CryptoJS from 'crypto-js';
import type { EncryptionMetadata, ProgressCallback, EncryptedData } from './types';

class StreamingEncryption {
  private chunkSize: number;
  private secretKey: string;

  constructor(chunkSize: number = 1024 * 1024) {
    // Default 1MB chunks
    this.chunkSize = chunkSize;
    this.secretKey =
      process.env.NEXT_PUBLIC_ENCRYPTION_KEY || 'default-secret-key';

    if (!process.env.NEXT_PUBLIC_ENCRYPTION_KEY) {
      console.warn(
        'NEXT_PUBLIC_ENCRYPTION_KEY not set, using default key (not secure!)'
      );
    }
  }

  /**
   * Optimize chunk size based on file size
   */
  getOptimalChunkSize(fileSize: number): number {
    if (fileSize < 10 * 1024 * 1024) return 512 * 1024; // 512KB for files < 10MB
    if (fileSize < 100 * 1024 * 1024) return 1024 * 1024; // 1MB for files < 100MB
    if (fileSize < 1024 * 1024 * 1024) return 2 * 1024 * 1024; // 2MB for files < 1GB
    return 4 * 1024 * 1024; // 4MB for very large files
  }

  /**
   * Main streaming encryption function
   */
  async encryptFileStream(
    file: File,
    progressCallback: ProgressCallback | null = null
  ): Promise<Uint8Array> {
    const optimalChunkSize = this.getOptimalChunkSize(file.size);
    const totalChunks = Math.ceil(file.size / optimalChunkSize);
    const encryptedChunks: string[] = [];

    for (let i = 0; i < totalChunks; i++) {
      try {
        // Calculate chunk boundaries
        const start = i * optimalChunkSize;
        const end = Math.min(start + optimalChunkSize, file.size);

        // Read only this chunk (not entire file)
        const chunk = file.slice(start, end);
        const arrayBuffer = await chunk.arrayBuffer();

        // Encrypt this chunk
        const encryptedChunk = await this.encryptChunk(arrayBuffer);
        encryptedChunks.push(encryptedChunk);

        // Update progress
        const progress = ((i + 1) / totalChunks) * 100;
        if (progressCallback) {
          progressCallback(progress);
        }

        // Yield control back to browser to keep UI responsive
        await this.sleep(5);

        // Log progress every 10 chunks for debugging
        if ((i + 1) % 10 === 0 || i === totalChunks - 1) {
          console.log(
            `Encrypted chunk ${i + 1}/${totalChunks} (${progress.toFixed(1)}%)`
          );
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown error';
        console.error(`Error encrypting chunk ${i + 1}:`, error);
        throw new Error(`Encryption failed at chunk ${i + 1}: ${errorMessage}`);
      }
    }

    // Combine all encrypted chunks with separator
    const combinedData = encryptedChunks.join('|CHUNK_SEPARATOR|');

    // Create metadata for reconstruction
    const metadata: EncryptionMetadata = {
      totalChunks: totalChunks,
      originalSize: file.size,
      chunkSize: optimalChunkSize,
      encrypted: true,
    };

    // Combine metadata and data
    const finalData =
      JSON.stringify(metadata) + '|METADATA_SEPARATOR|' + combinedData;

    // Convert string to Uint8Array using TextEncoder (browser-compatible)
    const encoder = new TextEncoder();
    return encoder.encode(finalData);
  }

  /**
   * Encrypt a single chunk
   */
  private async encryptChunk(arrayBuffer: ArrayBuffer): Promise<string> {
    try {
      // Convert ArrayBuffer to Uint8Array
      const uint8Array = new Uint8Array(arrayBuffer);

      // Convert to CryptoJS WordArray
      const wordArray = CryptoJS.lib.WordArray.create(uint8Array);

      // Encrypt the chunk
      const encrypted = CryptoJS.AES.encrypt(wordArray, this.secretKey);

      // Return as string
      return encrypted.toString();
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      console.error('Chunk encryption error:', error);
      throw new Error(`Chunk encryption failed: ${errorMessage}`);
    }
  }

  /**
   * Decrypt streamed file
   */
  async decryptFileStream(
    encryptedData: EncryptedData,
    progressCallback: ProgressCallback | null = null
  ): Promise<Uint8Array> {
    try {
      // Convert back to string if it's Uint8Array
      let dataString: string;
      if (encryptedData instanceof Uint8Array) {
        // Convert Uint8Array to string using TextDecoder (browser-compatible)
        const decoder = new TextDecoder();
        dataString = decoder.decode(encryptedData);
      } else {
        dataString = encryptedData;
      }

      // Split metadata and data
      const [metadataStr, chunksData] = dataString.split('|METADATA_SEPARATOR|');
      const metadata: EncryptionMetadata = JSON.parse(metadataStr);

      // Split into chunks
      const encryptedChunks = chunksData.split('|CHUNK_SEPARATOR|');
      const decryptedChunks: Uint8Array[] = [];

      for (let i = 0; i < encryptedChunks.length; i++) {
        try {
          // Decrypt chunk
          const decrypted = CryptoJS.AES.decrypt(
            encryptedChunks[i],
            this.secretKey
          );
          const decryptedArray = this.wordArrayToUint8Array(decrypted);
          decryptedChunks.push(decryptedArray);

          // Update progress
          const progress = ((i + 1) / encryptedChunks.length) * 100;
          if (progressCallback) {
            progressCallback(progress);
          }

          // Yield control
          await this.sleep(5);

          if ((i + 1) % 10 === 0 || i === encryptedChunks.length - 1) {
            console.log(
              `Decrypted chunk ${i + 1}/${encryptedChunks.length} (${progress.toFixed(1)}%)`
            );
          }
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Unknown error';
          console.error(`Error decrypting chunk ${i + 1}:`, error);
          throw new Error(`Decryption failed at chunk ${i + 1}: ${errorMessage}`);
        }
      }

      // Combine all decrypted chunks efficiently
      const result = this.combineChunks(decryptedChunks);
      return result;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error';
      console.error('Stream decryption error:', error);
      throw new Error(`Stream decryption failed: ${errorMessage}`);
    }
  }

  /**
   * Convert CryptoJS WordArray to Uint8Array
   */
  private wordArrayToUint8Array(wordArray: CryptoJS.lib.WordArray): Uint8Array {
    const words = wordArray.words;
    const sigBytes = wordArray.sigBytes;
    const u8 = new Uint8Array(sigBytes);

    for (let i = 0; i < sigBytes; i++) {
      u8[i] = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
    }

    return u8;
  }

  /**
   * Efficiently combine chunks into single Uint8Array
   */
  private combineChunks(chunks: Uint8Array[]): Uint8Array {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;

    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  /**
   * Non-blocking delay to keep UI responsive
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Check if file should use streaming (based on size)
   */
  shouldUseStreaming(fileSize: number): boolean {
    const STREAMING_THRESHOLD = 5 * 1024 * 1024; // 5MB
    return fileSize > STREAMING_THRESHOLD;
  }

  /**
   * Get memory-safe chunk count estimation
   */
  getChunkCount(fileSize: number): number {
    const chunkSize = this.getOptimalChunkSize(fileSize);
    return Math.ceil(fileSize / chunkSize);
  }
}

export default StreamingEncryption;
