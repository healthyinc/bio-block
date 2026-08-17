import { BrowserProvider, Contract } from 'ethers';
import ABI from './abi/AnalyticsRegistry.json';
import {
  ANALYTICS_REGISTRY_ADDRESS,
  SEPOLIA_CHAIN_ID,
  SEPOLIA_CHAIN_ID_HEX,
  EIP712_DOMAIN,
  EIP712_TYPES,
} from './constants';

export async function connectWallet() {
  if (!window.ethereum) {
    throw new Error('MetaMask is not installed. Please install it to continue.');
  }

  const provider = new BrowserProvider(window.ethereum);

  // Ensure Sepolia network
  const network = await provider.getNetwork();
  if (Number(network.chainId) !== SEPOLIA_CHAIN_ID) {
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: SEPOLIA_CHAIN_ID_HEX }],
      });
    } catch (switchErr) {
      // 4902: chain not added yet
      if (switchErr.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: SEPOLIA_CHAIN_ID_HEX,
            chainName: 'Sepolia Testnet',
            nativeCurrency: { name: 'Sepolia ETH', symbol: 'ETH', decimals: 18 },
            rpcUrls: ['https://rpc.sepolia.org'],
            blockExplorerUrls: ['https://sepolia.etherscan.io'],
          }],
        });
      } else {
        throw new Error('Please switch to the Sepolia testnet in MetaMask.');
      }
    }
  }

  const signer = await provider.getSigner();
  const address = await signer.getAddress();

  return { address, provider, signer };
}


export function truncateAddress(addr) {
  if (!addr) return '';
  return `${addr.slice(0, 6)}…${addr.slice(-4)}`;
}

// sign EIP-712 typed data
export async function signEIP712Request(signer, { datasetCID, requestHash }) {
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = Math.floor(Math.random() * 1_000_000_000);

  const value = {
    datasetCID,
    timestamp,
    nonce,
    requestHash,
  };

  const signature = await signer.signTypedData(EIP712_DOMAIN, EIP712_TYPES, value);

  return { signature, timestamp, nonce };
}

function _getRegistryContract(provider) {
  return new Contract(ANALYTICS_REGISTRY_ADDRESS, ABI, provider);
}

export async function getAnalyticsCount(provider, walletAddress) {
  const contract = _getRegistryContract(provider);
  const count = await contract.getAnalyticsCount(walletAddress);
  return Number(count);
}


export async function getAnalyticsForAddress(provider, walletAddress, offset = 0, limit = 20) {
  const contract = _getRegistryContract(provider);
  const records = await contract.getAnalyticsForAddress(walletAddress, offset, limit);

  return records.map((r) => ({
    sourceCID: r.sourceCID,
    resultCID: r.resultCID,
    analysisType: r.analysisType,
    timestamp: Number(r.timestamp),
  }));
}


export async function getAnalyticsForDataset(provider, sourceCID) {
  const contract = _getRegistryContract(provider);
  return await contract.getAnalyticsForDataset(sourceCID);
}
