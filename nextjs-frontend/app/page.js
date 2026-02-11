"use client";

import { useState } from "react";
import {
  Wallet,
  Search,
  Upload,
  User,
  ChevronDown,
  Shield,
  Database,
  Globe,
  Zap,
  BarChart3,
} from "lucide-react";
import SearchData from "../components/SearchData";
import UploadData from "../components/UploadData";
import Dashboard from "../components/Dashboard";
import Header from "../components/Header";
import HeroSection from "../components/HeroSection";

export default function Home() {
  const [isWalletConnected, setIsWalletConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState("");
  const [fullWalletAddress, setFullWalletAddress] = useState("");
  const [currentView, setCurrentView] = useState("main");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleWalletConnect = async () => {
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
        setFullWalletAddress(address);
        setIsWalletConnected(true);
      } else {
        alert("Please install MetaMask or another Web3 wallet");
      }
    } catch (error) {
      console.error("Wallet connection failed:", error);
      if (error.code === 4001) {
        alert("Connection rejected by user");
      }
    }
  };

  const handleDisconnect = () => {
    setIsWalletConnected(false);
    setWalletAddress("");
    setFullWalletAddress("");
    setIsDropdownOpen(false);
    setCurrentView("main");
  };

  const handleSearch = () => {
    setCurrentView("search");
  };

  const handleUpload = () => {
    setCurrentView("upload");
  };

  const handleDashboard = () => {
    setCurrentView("dashboard");
    setIsDropdownOpen(false);
  };

  const handleBackToMain = () => {
    setCurrentView("main");
  };

  if (currentView === "search") {
    return <SearchData onBack={handleBackToMain} />;
  }

  if (currentView === "upload") {
    return (
      <UploadData
        onBack={handleBackToMain}
        isWalletConnected={isWalletConnected}
        walletAddress={fullWalletAddress}
        onWalletConnect={handleWalletConnect}
      />
    );
  }

  if (currentView === "dashboard") {
    return (
      <Dashboard
        onBack={handleBackToMain}
        isWalletConnected={isWalletConnected}
        walletAddress={fullWalletAddress}
      />
    );
  }

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
          <div
            className={`border-l-4 p-4 rounded ${backendStatus.connected ? "bg-green-50 border-green-500" : "bg-yellow-50 border-yellow-500"}`}
          >
            <p className={backendStatus.connected ? "text-green-700" : "text-yellow-700"}>
              <strong>Backend Status:</strong> {backendStatus.message}
            </p>
          </div>
        </div>
      )}

      <main className="container mx-auto px-4 py-12">
        {!isWalletConnected ? (
          <HeroSection />
        ) : (
          <div className="max-w-6xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Welcome to Bio-Block Platform
              </h2>
              <p className="text-gray-600 mb-6">
                Your wallet is connected. Choose an action below to get started.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Search Documents */}
              <button
                onClick={() => router.push("/search")}
                className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-all hover:-translate-y-1 text-left"
              >
                <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
                  <Search size={32} className="text-blue-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-3">Search</h3>
                <p className="text-gray-600 mb-4">
                  Search and purchase medical datasets with advanced filters
                </p>
                <span className="text-blue-600 font-medium inline-flex items-center gap-2">
                  Browse Documents →
                </span>
              </button>

              {/* Upload Data */}
              <button
                onClick={() => router.push("/upload")}
                className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-all hover:-translate-y-1 text-left"
              >
                <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
                  <Upload size={32} className="text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-3">Upload</h3>
                <p className="text-gray-600 mb-4">
                  Upload your medical data securely with end-to-end encryption
                </p>
                <span className="text-green-600 font-medium inline-flex items-center gap-2">
                  Upload Document →
                </span>
              </button>

              {/* Dashboard */}
              <button
                onClick={() => router.push("/dashboard")}
                className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-all hover:-translate-y-1 text-left"
              >
                <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
                  <BarChart3 size={32} className="text-purple-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-3">Dashboard</h3>
                <p className="text-gray-600 mb-4">
                  View your earnings, documents, and manage your data
                </p>
                <span className="text-purple-600 font-medium inline-flex items-center gap-2">
                  View Dashboard →
                </span>
              </button>
            </div>

            {/* Features Info */}
            <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Phase 2 Features Completed ✓
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">Full TypeScript migration</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">Dashboard component</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">Upload component</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">Search component</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">App Router setup</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold">✓</span>
                  <span className="text-gray-700">Client/Server separation</span>
                </div>
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
