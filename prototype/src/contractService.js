import { ethers } from 'ethers';

const CONTRACT_ADDRESS = '0xd58de64aac08d5412b8020c7c61b215fec0c9644';

// ABI without on-chain document arrays — lists come from indexed events (#156).
const CONTRACT_ABI = [
  {
    inputs: [{ internalType: 'string', name: 'ipfsHash', type: 'string' }],
    name: 'purchaseDocument',
    outputs: [{ internalType: 'bool', name: '', type: 'bool' }],
    stateMutability: 'payable',
    type: 'function',
  },
  {
    inputs: [
      { internalType: 'string', name: 'ipfsHash', type: 'string' },
      { internalType: 'uint256', name: 'price', type: 'uint256' },
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
    inputs: [{ internalType: 'string', name: 'ipfsHash', type: 'string' }],
    name: 'deleteDocument',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [{ internalType: 'string', name: '', type: 'string' }],
    name: 'documentOwners',
    outputs: [{ internalType: 'address', name: '', type: 'address' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ internalType: 'string', name: '', type: 'string' }],
    name: 'documentPrices',
    outputs: [{ internalType: 'uint256', name: '', type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ internalType: 'address', name: '', type: 'address' }],
    name: 'earnings',
    outputs: [{ internalType: 'uint256', name: '', type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: 'address', name: 'owner', type: 'address' },
      { indexed: false, internalType: 'string', name: 'ipfsHash', type: 'string' },
      { indexed: false, internalType: 'uint256', name: 'price', type: 'uint256' },
    ],
    name: 'DocumentStored',
    type: 'event',
  },
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: 'address', name: 'owner', type: 'address' },
      { indexed: false, internalType: 'string', name: 'ipfsHash', type: 'string' },
    ],
    name: 'DocumentDeleted',
    type: 'event',
  },
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: 'address', name: 'buyer', type: 'address' },
      { indexed: true, internalType: 'address', name: 'owner', type: 'address' },
      { indexed: false, internalType: 'string', name: 'ipfsHash', type: 'string' },
      { indexed: false, internalType: 'uint256', name: 'amount', type: 'uint256' },
    ],
    name: 'DocumentPurchased',
    type: 'event',
  },
  {
    anonymous: false,
    inputs: [
      { indexed: true, internalType: 'address', name: 'owner', type: 'address' },
      { indexed: false, internalType: 'uint256', name: 'amount', type: 'uint256' },
    ],
    name: 'EarningsWithdrawn',
    type: 'event',
  },
];

const getContract = async (withSigner = false) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  if (!withSigner) {
    return { provider, contract: new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider) };
  }
  const signer = await provider.getSigner();
  return { provider, signer, contract: new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer) };
};

export const storeDocumentHash = async (ipfsHash, priceInEth) => {
  const { contract } = await getContract(true);
  const priceInWei = ethers.parseEther(priceInEth.toString());
  const tx = await contract.storeDocument(ipfsHash, priceInWei);
  await tx.wait();
  console.log(`Transaction successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const purchaseDocument = async (ipfsHash, priceInEth) => {
  const { contract } = await getContract(true);
  const priceInWei = ethers.parseEther(priceInEth.toString());
  const tx = await contract.purchaseDocument(ipfsHash, { value: priceInWei });
  await tx.wait();
  console.log(`Purchase successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const withdrawEarnings = async () => {
  const { contract } = await getContract(true);
  const tx = await contract.withdrawEarnings();
  await tx.wait();
  console.log(`Withdrawal successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const getDocumentPrice = async (ipfsHash) => {
  const { contract } = await getContract(false);
  const priceInWei = await contract.documentPrices(ipfsHash);
  return ethers.formatEther(priceInWei);
};

/**
 * Rebuild the caller's document list from indexed event logs.
 * Stored hashes minus later DocumentDeleted events for the same owner/hash.
 */
export const getMyDocuments = async () => {
  const { signer, contract } = await getContract(true);
  const owner = await signer.getAddress();

  const stored = await contract.queryFilter(contract.filters.DocumentStored(owner));
  const deleted = await contract.queryFilter(contract.filters.DocumentDeleted(owner));

  const active = new Set();
  // Apply logs in block/log order so deletes after stores win.
  const timeline = [...stored, ...deleted].sort((a, b) => {
    if (a.blockNumber !== b.blockNumber) return a.blockNumber - b.blockNumber;
    return a.index - b.index;
  });

  for (const log of timeline) {
    const hash = log.args.ipfsHash;
    const name = log.eventName || log.fragment?.name;
    if (name === 'DocumentStored') {
      active.add(hash);
    } else if (name === 'DocumentDeleted') {
      active.delete(hash);
    }
  }

  // Drop hashes whose on-chain ownership was cleared without a matching delete log.
  const verified = [];
  for (const hash of active) {
    const currentOwner = await contract.documentOwners(hash);
    if (currentOwner.toLowerCase() === owner.toLowerCase()) {
      verified.push(hash);
    }
  }

  return verified;
};

export const getEarnings = async (address) => {
  const { contract } = await getContract(false);
  const earningsInWei = await contract.earnings(address);
  return ethers.formatEther(earningsInWei);
};

export const deleteDocument = async (ipfsHash) => {
  const { contract } = await getContract(true);
  const tx = await contract.deleteDocument(ipfsHash);
  await tx.wait();
  console.log(`Delete successful with hash: ${tx.hash}`);
  return tx.hash;
};
