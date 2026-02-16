import { ethers, BrowserProvider, Contract } from 'ethers';
import type { ContractABI } from './types';

const CONTRACT_ADDRESS = '0xd58de64aac08d5412b8020c7c61b215fec0c9644';

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
    ],
    name: 'storeDocument',
    outputs: [],
    stateMutability: 'nonpayable',
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
 * Store document hash and price on blockchain
 * @param ipfsHash - IPFS hash of the encrypted document
 * @param priceInEth - Price in ETH
 * @returns Transaction hash
 */
export const storeDocumentHash = async (
  ipfsHash: string,
  priceInEth: number | string
): Promise<string> => {
  const contract = await getContract(true);
  const priceInWei = ethers.parseEther(priceInEth.toString());

  const tx = await contract.storeDocument(ipfsHash, priceInWei);
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
