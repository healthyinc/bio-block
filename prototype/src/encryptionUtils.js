// encryptionUtils.js
import CryptoJS from "crypto-js";

export const generateDocumentKey = () => {
  // Generate a random 256-bit (32 byte) key and return as hex string
  return CryptoJS.lib.WordArray.random(32).toString();
};

export const encryptFile = (fileBuffer, encryptionKey) => {
  if (!encryptionKey) throw new Error("Encryption key is required");
  const wordArray = CryptoJS.lib.WordArray.create(fileBuffer);
  const encrypted = CryptoJS.AES.encrypt(wordArray, ENCRYPTION_KEY).toString();
  return new Blob([encrypted], { type: "application/octet-stream" });
};

export const decryptFile = (encryptedData, encryptionKey) => {
  if (!encryptionKey) throw new Error("Decryption key is required");
  const decrypted = CryptoJS.AES.decrypt(encryptedData, encryptionKey);
  return decrypted.toString(CryptoJS.enc.Base64);
};
