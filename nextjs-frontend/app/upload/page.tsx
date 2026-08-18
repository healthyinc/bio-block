"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import UploadData from '../../components/UploadData';

export default function UploadPage() {
  const router = useRouter();
  const [isWalletConnected, setIsWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState('');
  const [fullWalletAddress, setFullWalletAddress] = useState('');

  // Check wallet connection on mount
  useEffect(() => {
    const checkWalletConnection = async () => {
      if (typeof window.ethereum !== 'undefined') {
        try {
          const accounts = await window.ethereum.request({
            method: 'eth_accounts',
          }) as string[];
          if (accounts.length > 0) {
            const address = accounts[0];
            const shortAddress = `${address.slice(0, 6)}...${address.slice(-4)}`;
            setWalletAddress(shortAddress);
            setFullWalletAddress(address);
            setIsWalletConnected(true);
          }
        } catch (error) {
          console.error('Error checking wallet connection:', error);
        }
      }
    };
    checkWalletConnection();
  }, []);

  const handleBack = () => {
    router.push('/');
  };

  const handleWalletConnect = async () => {
    try {
      if (typeof window.ethereum !== 'undefined') {
        await window.ethereum.request({
          method: 'wallet_requestPermissions',
          params: [{ eth_accounts: {} }],
        });

        const accounts = await window.ethereum.request({
          method: 'eth_requestAccounts',
        }) as string[];

        const address = accounts[0];
        const shortAddress = `${address.slice(0, 6)}...${address.slice(-4)}`;
        setWalletAddress(shortAddress);
        setFullWalletAddress(address);
        setIsWalletConnected(true);
      } else {
        alert('Please install MetaMask or another Web3 wallet');
      }
    } catch (error) {
      console.error('Wallet connection failed:', error);
    }
  };

  return (
    <UploadData
      onBack={handleBack}
      isWalletConnected={isWalletConnected}
      walletAddress={fullWalletAddress}
      onWalletConnect={handleWalletConnect}
    />
  );
}
