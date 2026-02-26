"use client";

import { useRouter } from 'next/navigation';
import { useAccount } from 'wagmi';
import Dashboard from '../../components/Dashboard';

export default function DashboardPage() {
  const router = useRouter();
  const { address, isConnected } = useAccount();

  const handleBack = () => {
    router.push('/');
  };

  return (
    <Dashboard
      onBack={handleBack}
      isWalletConnected={isConnected}
      walletAddress={address || ''}
    />
  );
}
