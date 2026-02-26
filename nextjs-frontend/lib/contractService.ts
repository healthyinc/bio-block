import { parseEther, formatEther, type Address } from 'viem';
import { readContract, writeContract, waitForTransactionReceipt, getAccount, type Config } from '@wagmi/core';
import { CONTRACT_ADDRESS, CONTRACT_ABI, CONTRACT_CHAIN_ID, config } from './wagmi';

// Helper to get typed config - wagmi types are complex, this simplifies usage
const getConfig = () => config as unknown as Config;

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
  const priceInWei = parseEther(priceInEth.toString());

  const hash = await writeContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'storeDocument',
    args: [ipfsHash, priceInWei],
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof writeContract>[1]);

  await waitForTransactionReceipt(getConfig(), { hash, chainId: CONTRACT_CHAIN_ID });
  return hash;
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
  const priceInWei = parseEther(priceInEth.toString());

  const hash = await writeContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'purchaseDocument',
    args: [ipfsHash],
    value: priceInWei,
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof writeContract>[1]);

  await waitForTransactionReceipt(getConfig(), { hash, chainId: CONTRACT_CHAIN_ID });
  return hash;
};

/**
 * Withdraw earnings from the contract
 * @returns Transaction hash
 */
export const withdrawEarnings = async (): Promise<string> => {
  const hash = await writeContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'withdrawEarnings',
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof writeContract>[1]);

  await waitForTransactionReceipt(getConfig(), { hash, chainId: CONTRACT_CHAIN_ID });
  return hash;
};

/**
 * Get the price of a document
 * @param ipfsHash - IPFS hash of the document
 * @returns Price in ETH as string
 */
export const getDocumentPrice = async (ipfsHash: string): Promise<string> => {
  const priceInWei = await readContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'documentPrices',
    args: [ipfsHash],
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof readContract>[1]);

  return formatEther(priceInWei as bigint);
};

/**
 * Get all documents uploaded by the current user
 * @returns Array of IPFS hashes
 */
export const getMyDocuments = async (): Promise<string[]> => {
  const account = getAccount(getConfig());
  if (!account.address) {
    throw new Error('Wallet not connected');
  }

  const documents = await readContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'getDocuments',
    args: [account.address],
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof readContract>[1]);

  return documents as string[];
};

/**
 * Get earnings for a specific address
 * @param address - Ethereum address
 * @returns Earnings in ETH as string
 */
export const getEarnings = async (address: string): Promise<string> => {
  const earningsInWei = await readContract(getConfig(), {
    address: CONTRACT_ADDRESS,
    abi: CONTRACT_ABI,
    functionName: 'earnings',
    args: [address as Address],
    chainId: CONTRACT_CHAIN_ID,
  } as unknown as Parameters<typeof readContract>[1]);

  return formatEther(earningsInWei as bigint);
};

/**
 * Get contract address
 * @returns Contract address
 */
export const getContractAddress = (): string => {
  return CONTRACT_ADDRESS;
};

/**
 * Get current connected account from wagmi
 * @returns Account info or undefined
 */
export const getConnectedAccount = () => {
  return getAccount(getConfig());
};

/**
 * Get the chain ID where the contract is deployed
 * @returns Chain ID
 */
export const getContractChainId = (): number => {
  return CONTRACT_CHAIN_ID;
};
