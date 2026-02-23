import { ethers } from "ethers";

const CONTRACT_ADDRESS = "0xd58de64aac08d5412b8020c7c61b215fec0c9644";
const CONTRACT_ABI = [
  {
    inputs: [
      {
        internalType: "string",
        name: "ipfsHash",
        type: "string",
      },
    ],
    name: "purchaseDocument",
    outputs: [
      {
        internalType: "bool",
        name: "",
        type: "bool",
      },
    ],
    stateMutability: "payable",
    type: "function",
  },
  {
    inputs: [
      {
        internalType: "string",
        name: "ipfsHash",
        type: "string",
      },
      {
        internalType: "uint256",
        name: "price",
        type: "uint256",
      },
    ],
    name: "storeDocument",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [],
    name: "withdrawEarnings",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [
      {
        internalType: "string",
        name: "",
        type: "string",
      },
    ],
    name: "documentOwners",
    outputs: [
      {
        internalType: "address",
        name: "",
        type: "address",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      {
        internalType: "string",
        name: "",
        type: "string",
      },
    ],
    name: "documentPrices",
    outputs: [
      {
        internalType: "uint256",
        name: "",
        type: "uint256",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      {
        internalType: "address",
        name: "",
        type: "address",
      },
    ],
    name: "earnings",
    outputs: [
      {
        internalType: "uint256",
        name: "",
        type: "uint256",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [
      {
        internalType: "address",
        name: "user",
        type: "address",
      },
    ],
    name: "getDocuments",
    outputs: [
      {
        internalType: "string[]",
        name: "",
        type: "string[]",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [],
    name: "getMyDocuments",
    outputs: [
      {
        internalType: "string[]",
        name: "",
        type: "string[]",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
];

export const storeDocumentHash = async (ipfsHash, priceInEth) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

  const priceInWei = ethers.parseEther(priceInEth.toString());
  const tx = await contract.storeDocument(ipfsHash, priceInWei);
  await tx.wait();
  console.log(`Transaction successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const purchaseDocument = async (ipfsHash, priceInEth) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

  const priceInWei = ethers.parseEther(priceInEth.toString());
  const tx = await contract.purchaseDocument(ipfsHash, { value: priceInWei });
  await tx.wait();
  console.log(`Purchase successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const withdrawEarnings = async () => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

  const tx = await contract.withdrawEarnings();
  await tx.wait();
  console.log(`Withdrawal successful with hash: ${tx.hash}`);
  return tx.hash;
};

export const getDocumentPrice = async (ipfsHash) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);

  const priceInWei = await contract.documentPrices(ipfsHash);
  return ethers.formatEther(priceInWei);
};

export const getMyDocuments = async () => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const signer = await provider.getSigner();
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

  const documents = await contract.getMyDocuments();
  return documents;
};

export const getEarnings = async (address) => {
  const provider = new ethers.BrowserProvider(window.ethereum);
  const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);

  const earningsInWei = await contract.earnings(address);
  return ethers.formatEther(earningsInWei);
};
