import CryptoJS from 'crypto-js';
import type { FileBuffer, EncryptedData } from './types';



export const generateDocumentKey = (): string => {
  return CryptoJS.lib.WordArray.random(32).toString();
};

/**
 * Encrypts a file buffer using AES encryption
 * @param fileBuffer - The file buffer to encrypt (ArrayBuffer or Uint8Array)
 * @param encryptionKey - The secret key to encrypt with
 * @returns Encrypted blob
 */
export const encryptFile = (fileBuffer: FileBuffer, encryptionKey: string): Blob => {
  if (!encryptionKey) {
    throw new Error('Encryption key is not configured');
  }

  const wordArray = CryptoJS.lib.WordArray.create(
    fileBuffer instanceof ArrayBuffer ? new Uint8Array(fileBuffer) : fileBuffer
  );
  const encrypted = CryptoJS.AES.encrypt(wordArray, encryptionKey).toString();
  return new Blob([encrypted], { type: 'application/octet-stream' });
};

/**
 * Decrypts encrypted data using AES decryption
 * @param encryptedData - The encrypted data (string or Uint8Array)
 * @param encryptionKey - The secret key to decrypt with
 * @returns Decrypted data as Base64 string
 */
export const decryptFile = (encryptedData: EncryptedData, encryptionKey: string): string => {
  if (!encryptionKey) {
    throw new Error('Encryption key is not configured');
  }

  const dataString =
    typeof encryptedData === 'string'
      ? encryptedData
      : new TextDecoder().decode(encryptedData);

  const decrypted = CryptoJS.AES.decrypt(dataString, encryptionKey);
  return decrypted.toString(CryptoJS.enc.Base64);
};


