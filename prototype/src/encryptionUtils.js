// encryptionUtils.js
import CryptoJS from "crypto-js";

const ENCRYPTION_KEY = process.env.REACT_APP_ENCRYPTION_KEY;

export const encryptFile = (fileBuffer) => {
  const wordArray = CryptoJS.lib.WordArray.create(fileBuffer);
  const encrypted = CryptoJS.AES.encrypt(wordArray, ENCRYPTION_KEY).toString();
  return new Blob([encrypted], { type: "application/octet-stream" });
};

export const decryptFile = (encryptedData) => {
  const decrypted = CryptoJS.AES.decrypt(encryptedData, ENCRYPTION_KEY);
  return decrypted.toString(CryptoJS.enc.Base64);
};
