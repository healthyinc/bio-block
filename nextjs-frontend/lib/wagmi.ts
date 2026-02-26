import { getDefaultConfig } from '@rainbow-me/rainbowkit';
import { http } from 'wagmi';
import { mainnet, sepolia, polygon, polygonAmoy } from 'wagmi/chains';

// Project ID from WalletConnect Cloud (get one at https://cloud.walletconnect.com)
const projectId = process.env.NEXT_PUBLIC_WALLET_CONNECT_PROJECT_ID || 'YOUR_PROJECT_ID';

export const config = getDefaultConfig({
  appName: 'Bio-Block',
  projectId,
  chains: [mainnet, sepolia, polygon, polygonAmoy],
  transports: {
    [mainnet.id]: http(),
    [sepolia.id]: http(),
    [polygon.id]: http(),
    [polygonAmoy.id]: http(),
  },
  ssr: true,
});

// Contract configuration
export const CONTRACT_ADDRESS = '0xd58de64aac08d5412b8020c7c61b215fec0c9644' as const;

// The chain where the contract is deployed
export const CONTRACT_CHAIN_ID = sepolia.id;
export const CONTRACT_CHAIN = sepolia;

export const CONTRACT_ABI = [
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
] as const;
