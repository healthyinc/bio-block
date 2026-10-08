// deployed contracts
export const ANALYTICS_REGISTRY_ADDRESS = '0x8D7eE1371509CA824ad85f146dFBE0875Dd60610';
export const DOCUMENT_STORAGE_ADDRESS = '0xd58de64aac08d5412b8020c7c61b215fec0c9644';

// sepolia
export const SEPOLIA_CHAIN_ID = 11155111;
export const SEPOLIA_CHAIN_ID_HEX = '0xaa36a7';
export const SEPOLIA_ETHERSCAN = 'https://sepolia.etherscan.io';
export const PINATA_GATEWAY = 'https://gateway.pinata.cloud/ipfs';

// EIP-712 schema (must match eip712.py)
export const EIP712_DOMAIN = {
  name: 'BioBlockAnalytics',
  version: '1',
  verifyingContract: DOCUMENT_STORAGE_ADDRESS,
};

export const EIP712_TYPES = {
  AnalyticsRequest: [
    { name: 'datasetCID', type: 'string' },
    { name: 'timestamp', type: 'uint256' },
    { name: 'nonce', type: 'uint256' },
    { name: 'requestHash', type: 'string' },
  ],
};
