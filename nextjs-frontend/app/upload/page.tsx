"use client";

import { useRouter } from 'next/navigation';
import { useAccount } from 'wagmi';
import UploadData from '../../components/UploadData';

export default function UploadPage() {
  const router = useRouter();
  const { address, isConnected } = useAccount();

  const handleBack = () => {
    router.push('/');
  };

  return (
    <UploadData
      onBack={handleBack}
      isWalletConnected={isConnected}
      walletAddress={address || ''}
    />
  );
}
