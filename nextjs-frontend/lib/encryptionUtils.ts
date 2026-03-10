import CryptoJS from 'crypto-js';
import type { FileBuffer, EncryptedData } from './types';

const ENCRYPTION_KEY = process.env.NEXT_PUBLIC_ENCRYPTION_KEY;

if (!ENCRYPTION_KEY) {
  console.warn('NEXT_PUBLIC_ENCRYPTION_KEY is not set. Encryption will not work properly.');
}

/**
 * Encrypts a file buffer using AES encryption
 * @param fileBuffer - The file buffer to encrypt (ArrayBuffer or Uint8Array)
 * @returns Encrypted blob
 */
export const encryptFile = (fileBuffer: FileBuffer): Blob => {
  if (!ENCRYPTION_KEY) {
    throw new Error('Encryption key is not configured');
  }

  const wordArray = CryptoJS.lib.WordArray.create(
    fileBuffer instanceof ArrayBuffer ? new Uint8Array(fileBuffer) : fileBuffer
  );
  const encrypted = CryptoJS.AES.encrypt(wordArray, ENCRYPTION_KEY).toString();
  return new Blob([encrypted], { type: 'application/octet-stream' });
};

/**
 * Decrypts encrypted data using AES decryption
 * @param encryptedData - The encrypted data (string or Uint8Array)
 * @returns Decrypted data as Base64 string
 */
export const decryptFile = (encryptedData: EncryptedData): string => {
  if (!ENCRYPTION_KEY) {
    throw new Error('Encryption key is not configured');
  }

  const dataString =
    typeof encryptedData === 'string'
      ? encryptedData
      : new TextDecoder().decode(encryptedData);

  const decrypted = CryptoJS.AES.decrypt(dataString, ENCRYPTION_KEY);
  return decrypted.toString(CryptoJS.enc.Base64);
};

/**
 * Validates if the encryption key is properly configured
 * @returns true if encryption key is set, false otherwise
 */
export const isEncryptionConfigured = (): boolean => {
  return !!ENCRYPTION_KEY && ENCRYPTION_KEY.length > 0;
};
