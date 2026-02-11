"use client";

import { Shield, Wallet } from "lucide-react";

interface HeaderProps {
  isWalletConnected: boolean;
  walletAddress: string;
  onConnect: () => void;
  onDisconnect: () => void;
}

export default function Header({ isWalletConnected, walletAddress, onConnect, onDisconnect }: HeaderProps) {
  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Shield className="w-8 h-8 text-indigo-600" />
          <h1 className="text-2xl font-bold text-gray-800">Bio-Block</h1>
        </div>

        {!isWalletConnected ? (
          <button
            onClick={onConnect}
            className="flex items-center space-x-2 bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <Wallet className="w-5 h-5" />
            <span>Connect Wallet</span>
          </button>
        ) : (
          <div className="flex items-center space-x-4">
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg font-mono text-sm">
              {walletAddress}
            </div>
            <button onClick={onDisconnect} className="text-red-600 hover:text-red-800 font-medium">
              Disconnect
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
