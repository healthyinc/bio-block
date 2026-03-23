export interface EncryptionMetadata {
  totalChunks: number;
  originalSize: number;
  chunkSize: number;
  encrypted: boolean;
}

export type ProgressCallback = (progress: number) => void;

export interface DocumentInfo {
  hash: string;
  price: string;
}

export interface TransactionResult {
  hash: string;
}

export interface EthereumProvider {
  request: (args: { method: string; params?: unknown[] }) => Promise<unknown>;
  on?: (event: string, callback: (...args: unknown[]) => void) => void;
  removeListener?: (event: string, callback: (...args: unknown[]) => void) => void;
}

declare global {
  interface Window {
    ethereum?: EthereumProvider;
  }
}

export interface ContractABI {
  inputs: Array<{ internalType: string; name: string; type: string }>;
  name?: string;
  outputs?: Array<{ internalType: string; name: string; type: string }>;
  stateMutability: string;
  type: string;
}

export type FileBuffer = ArrayBuffer | Uint8Array;
export type EncryptedData = string | Uint8Array;
