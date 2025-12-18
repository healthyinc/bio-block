"use client";

import { useState, useEffect } from "react";
import Header from "../components/Header";
import HeroSection from "../components/HeroSection";
import { checkBackendHealth } from "../lib/api";

export default function Home() {
  const [isWalletConnected, setIsWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState("");
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState({ checked: false, connected: false, message: "" });

  // Check backend connectivity on mount
  useEffect(() => {
    const verifyBackend = async () => {
      const result = await checkBackendHealth();
      setBackendStatus({
        checked: true,
        connected: result.success,
        message: result.message,
      });
    };
    verifyBackend();
  }, []);

  const handleWalletConnect = async () => {
    setError("");
    try {
      if (typeof window.ethereum !== "undefined") {
        await window.ethereum.request({
          method: "wallet_requestPermissions",
          params: [{ eth_accounts: {} }],
        });

        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });

        const address = accounts[0];
        const shortAddress = `${address.slice(0, 6)}...${address.slice(-4)}`;
        setWalletAddress(shortAddress);
        setIsWalletConnected(true);
      } else {
        setError("Please install MetaMask or another Web3 wallet");
      }
    } catch (error) {
      console.error("Wallet connection failed:", error);
      if (error.code === 4001) {
        setError("Connection rejected by user");
      } else {
        setError("Failed to connect wallet. Please try again.");
      }
    }
  };

  const handleDisconnect = () => {
    setIsWalletConnected(false);
    setWalletAddress("");
    setError("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header
        isWalletConnected={isWalletConnected}
        walletAddress={walletAddress}
        onConnect={handleWalletConnect}
        onDisconnect={handleDisconnect}
      />

      {error && (
        <div className="container mx-auto px-4 pt-4">
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
            <p className="text-red-700">{error}</p>
          </div>
        </div>
      )}

      {backendStatus.checked && (
        <div className="container mx-auto px-4 pt-4">
          <div className={`border-l-4 p-4 rounded ${backendStatus.connected ? 'bg-green-50 border-green-500' : 'bg-yellow-50 border-yellow-500'}`}>
            <p className={backendStatus.connected ? 'text-green-700' : 'text-yellow-700'}>
              <strong>Backend Status:</strong> {backendStatus.message}
            </p>
          </div>
        </div>
      )}

      <main className="container mx-auto px-4 py-12">
        {!isWalletConnected ? (
          <HeroSection />
        ) : (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-6">
                Welcome to Bio-Block Dashboard
              </h2>
              <p className="text-gray-600 mb-4">
                Your wallet is connected. Next PR will add full upload/download functionality.
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-blue-800">
                  <strong>✓ Phase 1 Complete:</strong> Next.js 15 setup with wallet connection
                </p>
                <p className="text-blue-700 mt-2">
                  <strong>Next:</strong> Full feature migration in PR #2
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gray-800 text-white py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>© 2025 Bio-Block - Hybrid Architecture Implementation</p>
          <p className="text-sm text-gray-400 mt-2">
            Client-side encryption + Backend IPFS + Blockchain storage
          </p>
        </div>
      </footer>
    </div>
  );
}
