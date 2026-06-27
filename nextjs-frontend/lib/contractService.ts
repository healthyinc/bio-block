import { ethers, BrowserProvider, Contract } from 'ethers';
import type { ContractABI, AnalyticsRecord } from './types';

const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_CONTRACT_ADDRESS!;
const ANALYTICS_REGISTRY_ADDRESS = process.env.NEXT_PUBLIC_ANALYTICS_REGISTRY_ADDRESS!;

const CONTRACT_ABI: ContractABI[] = [
  {
    inputs: [
      {
        internalType: 'string',
        name: 'ipfsHash',
        type: 'string',
      },
    ],
    name: 'purchaseDocument',
    outputs: [
      {
        internalType: 'bool',
        name: '',
        type: 'bool',
      },
    ],
    stateMutability: 'payable',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: 'ipfsHash',
        type: 'string',
      },
      {
        internalType: 'uint256',
        name: 'price',
        type: 'uint256',
      },
      {
        internalType: 'string',
        name: 'metadata',
        type: 'string',
      },
    ],
    name: 'storeDocument',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: 'ipfsHash',
        type: 'string',
      },
      {
        internalType: 'uint256',
        name: 'newPrice',
        type: 'uint256',
      },
      {
        internalType: 'string',
        name: 'newMetadata',
        type: 'string',
      },
    ],
    name: 'updateMetadata',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: 'ipfsHash',
        type: 'string',
      },
    ],
    name: 'deleteDocument',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: 'ipfsHash',
        type: 'string',
      },
    ],
    name: 'getMetadata',
    outputs: [
      {
        internalType: 'string',
        name: '',
        type: 'string',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [],
    name: 'withdrawEarnings',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: '',
        type: 'string',
      },
    ],
    name: 'documentOwners',
    outputs: [
      {
        internalType: 'address',
        name: '',
        type: 'address',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'string',
        name: '',
        type: 'string',
      },
    ],
    name: 'documentPrices',
    outputs: [
      {
        internalType: 'uint256',
        name: '',
        type: 'uint256',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'address',
        name: '',
        type: 'address',
      },
    ],
    name: 'earnings',
    outputs: [
      {
        internalType: 'uint256',
        name: '',
        type: 'uint256',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      {
        internalType: 'address',
        name: 'user',
        type: 'address',
      },
    ],
    name: 'getDocuments',
    outputs: [
      {
        internalType: 'string[]',
        name: '',
        type: 'string[]',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [],
    name: 'getMyDocuments',
    outputs: [
      {
        internalType: 'string[]',
        name: '',
        type: 'string[]',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
];

const ANALYTICS_REGISTRY_ABI = [
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: 'address', name: 'analyst', type: 'address' },
      { indexed: false, internalType: 'string', name: 'sourceCID', type: 'string' },
      { indexed: false, internalType: 'string', name: 'resultCID', type: 'string' },
      { indexed: false, internalType: 'string', name: 'analysisType', type: 'string' },
      { indexed: false, internalType: 'uint256', name: 'timestamp', type: 'uint256' },
    ],
    name: 'AnalyticsRegistered',
    type: 'event',
  },
  {
    inputs: [
      { internalType: 'string', name: 'sourceCID', type: 'string' },
    ],
    name: 'getAnalyticsForDataset',
    outputs: [
      { internalType: 'string[]', name: '', type: 'string[]' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      { internalType: 'uint256', name: 'offset', type: 'uint256' },
      { internalType: 'uint256', name: 'limit', type: 'uint256' },
    ],
    name: 'getMyAnalytics',
    outputs: [
      {
        components: [
          { internalType: 'string', name: 'sourceCID', type: 'string' },
          { internalType: 'string', name: 'resultCID', type: 'string' },
          { internalType: 'string', name: 'analysisType', type: 'string' },
          { internalType: 'uint256', name: 'timestamp', type: 'uint256' },
        ],
        internalType: 'struct AnalyticsRegistry.AnalyticsRecord[]',
        name: '',
        type: 'tuple[]',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [],
    name: 'getMyAnalyticsCount',
    outputs: [
      { internalType: 'uint256', name: '', type: 'uint256' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      { internalType: 'string', name: 'sourceCID', type: 'string' },
      { internalType: 'string', name: 'resultCID', type: 'string' },
      { internalType: 'string', name: 'analysisType', type: 'string' },
    ],
    name: 'registerAnalytics',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
];

/**
 * Get the provider and signer for contract interactions
 */
const getProviderAndSigner = async () => {
  if (typeof window === 'undefined' || !window.ethereum) {
    throw new Error('Ethereum provider not found. Please install MetaMask.');
  }

  const provider = new BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  return { provider, signer };
};

/**
 * Get the contract instance
 */
const getContract = async (withSigner = true): Promise<Contract> => {
  const { provider, signer } = await getProviderAndSigner();
  return new Contract(
    CONTRACT_ADDRESS,
    CONTRACT_ABI,
    withSigner ? signer : provider
  );
};

/**
 * Get the analytics registry contract instance
 */
const getAnalyticsContract = async (withSigner = true): Promise<Contract> => {
  const { provider, signer } = await getProviderAndSigner();
  return new Contract(
    ANALYTICS_REGISTRY_ADDRESS,
    ANALYTICS_REGISTRY_ABI,
    withSigner ? signer : provider
  );
};

/**
 * Store document hash, price and metadata on blockchain
 * @param ipfsHash - IPFS hash of the encrypted document
 * @param priceInEth - Price in ETH
 * @param metadata - Optional metadata string (e.g. tags, description)
 * @returns Transaction hash
 */
export const storeDocumentHash = async (
  ipfsHash: string,
  priceInEth: number | string,
  metadata: string = ''
): Promise<string> => {
  const contract = await getContract(true);
  const priceInWei = ethers.parseEther(priceInEth.toString());

  const tx = await contract.storeDocument(ipfsHash, priceInWei, metadata);
  await tx.wait();
  return tx.hash;
};

/**
 * Purchase a document from the blockchain
 * @param ipfsHash - IPFS hash of the document
 * @param priceInEth - Price in ETH
 * @returns Transaction hash
 */
export const purchaseDocument = async (
  ipfsHash: string,
  priceInEth: number | string
): Promise<string> => {
  const contract = await getContract(true);
  const priceInWei = ethers.parseEther(priceInEth.toString());

  const tx = await contract.purchaseDocument(ipfsHash, { value: priceInWei });
  await tx.wait();
  return tx.hash;
};

/**
 * Withdraw earnings from the contract
 * @returns Transaction hash
 */
export const withdrawEarnings = async (): Promise<string> => {
  const contract = await getContract(true);

  const tx = await contract.withdrawEarnings();
  await tx.wait();
  return tx.hash;
};

/**
 * Get the price of a document
 * @param ipfsHash - IPFS hash of the document
 * @returns Price in ETH as string
 */
export const getDocumentPrice = async (ipfsHash: string): Promise<string> => {
  const contract = await getContract(false);

  const priceInWei = await contract.documentPrices(ipfsHash);
  return ethers.formatEther(priceInWei);
};

/**
 * Get all documents uploaded by the current user
 * @returns Array of IPFS hashes
 */
export const getMyDocuments = async (): Promise<string[]> => {
  const contract = await getContract(true);

  const documents = await contract.getMyDocuments();
  return documents;
};

/**
 * Get earnings for a specific address
 * @param address - Ethereum address
 * @returns Earnings in ETH as string
 */
export const getEarnings = async (address: string): Promise<string> => {
  const contract = await getContract(false);

  const earningsInWei = await contract.earnings(address);
  return ethers.formatEther(earningsInWei);
};

/**
 * Get metadata for a document
 * @param ipfsHash - IPFS hash of the document
 * @returns Metadata string
 */
export const getMetadata = async (ipfsHash: string): Promise<string> => {
  const contract = await getContract(false);
  return contract.getMetadata(ipfsHash);
};

/**
 * Update document metadata and price (owner only)
 * @param ipfsHash - IPFS hash of the document
 * @param newPriceInEth - New price in ETH
 * @param newMetadata - New metadata string
 * @returns Transaction hash
 */
export const updateMetadata = async (
  ipfsHash: string,
  newPriceInEth: number | string,
  newMetadata: string
): Promise<string> => {
  const contract = await getContract(true);
  const newPriceInWei = ethers.parseEther(newPriceInEth.toString());

  const tx = await contract.updateMetadata(ipfsHash, newPriceInWei, newMetadata);
  await tx.wait();
  return tx.hash;
};

/**
 * Delete a document (owner only)
 * @param ipfsHash - IPFS hash of the document
 * @returns Transaction hash
 */
export const deleteDocument = async (ipfsHash: string): Promise<string> => {
  const contract = await getContract(true);

  const tx = await contract.deleteDocument(ipfsHash);
  await tx.wait();
  return tx.hash;
};

/**
 * Check if MetaMask is installed
 * @returns true if MetaMask is installed
 */
export const isMetaMaskInstalled = (): boolean => {
  return typeof window !== 'undefined' && typeof window.ethereum !== 'undefined';
};

/**
 * Get contract address
 * @returns Contract address
 */
export const getContractAddress = (): string => {
  return CONTRACT_ADDRESS;
};

/**
 * Get analytics registry contract address
 * @returns Analytics registry contract address
 */
export const getAnalyticsRegistryAddress = (): string => {
  return ANALYTICS_REGISTRY_ADDRESS;
};

/**
 * Register analytics result on blockchain
 * @param sourceCID - IPFS hash of the source document
 * @param resultCID - IPFS hash of the analytics result
 * @param analysisType - Type of analysis (descriptive, graphical, inferential)
 * @returns Transaction hash
 */
export const registerAnalytics = async (
  sourceCID: string,
  resultCID: string,
  analysisType: string
): Promise<string> => {
  const contract = await getAnalyticsContract(true);

  const tx = await contract.registerAnalytics(sourceCID, resultCID, analysisType);
  await tx.wait();
  return tx.hash;
};

/**
 * Get all analytics results for a specific dataset
 * @param sourceCID - IPFS hash of the source document
 * @returns Array of result CIDs
 */
export const getAnalyticsForDataset = async (sourceCID: string): Promise<string[]> => {
  const contract = await getAnalyticsContract(false);
  return contract.getAnalyticsForDataset(sourceCID);
};

/**
 * Get analytics results for current user (paginated)
 * @param offset - Pagination offset
 * @param limit - Max number of results to return
 * @returns Array of analytics records
 */
export const getMyAnalytics = async (offset: number, limit: number): Promise<AnalyticsRecord[]> => {
  // msg.sender-dependent — needs signer, not provider
  const contract = await getAnalyticsContract(true);
  const results = await contract.getMyAnalytics(offset, limit);

  return results.map((record: any) => ({
    sourceCID: record.sourceCID,
    resultCID: record.resultCID,
    analysisType: record.analysisType,
    timestamp: Number(record.timestamp),
  }));
};

/**
 * Get total count of analytics results for current user
 * @returns Number of analytics records
 */
export const getMyAnalyticsCount = async (): Promise<number> => {
  // msg.sender-dependent — needs signer, not provider
  const contract = await getAnalyticsContract(true);
  const count = await contract.getMyAnalyticsCount();
  return Number(count);
};
