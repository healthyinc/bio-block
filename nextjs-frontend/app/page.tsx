"use client";

import { useState } from "react";
import { Wallet, Search, Upload, User, ChevronDown, Shield, Database, Globe, Zap } from "lucide-react";
import SearchData from "../components/SearchData";
import UploadData from "../components/UploadData";
import Dashboard from "../components/Dashboard";

type ViewType = "main" | "search" | "upload" | "dashboard";

export default function Home() {
  const [isWalletConnected, setIsWalletConnected] = useState<boolean>(false);
  const [walletAddress, setWalletAddress] = useState<string>("");
  const [fullWalletAddress, setFullWalletAddress] = useState<string>("");
  const [currentView, setCurrentView] = useState<ViewType>("main");
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);

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
    } catch (error: unknown) {
      console.error("Wallet connection failed:", error);
      if (error && typeof error === 'object' && 'code' in error && (error as { code: number }).code === 4001) {
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Enhanced Header with Gradient */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-blue-600 to-indigo-600 h-2"></div>
      
      {/* Wallet Connection Section */}
      <div className="absolute top-6 right-6">
        {isWalletConnected ? (
          <div className="relative">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center gap-3 px-5 py-3 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-all duration-200 shadow-lg hover:shadow-xl backdrop-blur-sm"
            >
              <div className="w-9 h-9 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full flex items-center justify-center shadow-md">
                <User size={18} className="text-white" />
              </div>
              <span className="text-sm font-medium text-gray-700">{walletAddress}</span>
              <ChevronDown size={16} className={`text-gray-500 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`} />
            </button>

            {isDropdownOpen && (
              <div className="absolute right-0 mt-3 w-52 bg-white border border-gray-200 rounded-xl shadow-xl z-10 backdrop-blur-sm">
                <button
                  onClick={handleDashboard}
                  className="w-full text-left px-5 py-4 text-sm text-gray-700 hover:bg-blue-50 transition-colors border-b border-gray-100 rounded-t-xl font-medium"
                >
                  View My Profile
                </button>
                <button
                  onClick={handleDisconnect}
                  className="w-full text-left px-5 py-4 text-sm text-red-600 hover:bg-red-50 transition-colors rounded-b-xl font-medium"
                >
                  Disconnect Wallet
                </button>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={handleWalletConnect}
            className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            <Wallet size={18} />
            <span className="font-medium">Connect Wallet</span>
          </button>
        )}
      </div>

      {/* Main Content Container */}
      <div className="flex items-center justify-center min-h-screen p-6">
        <div className="w-full max-w-4xl">
          
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl mb-6 shadow-lg">
              <Shield size={32} className="text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
              Bio-Block
            </h1>
            <p className="text-xl text-gray-600 mb-2 max-w-2xl mx-auto leading-relaxed">
              Secure, decentralized document management powered by blockchain technology
            </p>
            <p className="text-gray-500 max-w-xl mx-auto">
              Upload, search, and manage your medical documents with enterprise-grade security and privacy
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4">
                <Database size={24} className="text-blue-600" />
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">IPFS Storage</h3>
              <p className="text-sm text-gray-600">Decentralized storage ensures your documents are always accessible and secure</p>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4">
                <Shield size={24} className="text-green-600" />
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">Blockchain Verified</h3>
              <p className="text-sm text-gray-600">Ethereum blockchain ensures document integrity and ownership verification</p>
            </div>
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-200 border border-gray-100">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-4">
                <Zap size={24} className="text-purple-600" />
              </div>
              <h3 className="font-semibold text-gray-800 mb-2">Smart Search</h3>
              <p className="text-sm text-gray-600">Advanced filtering and semantic search to find exactly what you need</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8 border border-gray-100">
            <div className="grid md:grid-cols-2 gap-6">
              <button
                onClick={handleSearch}
                className="group relative flex items-center justify-center gap-4 px-8 py-6 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                    <Search size={24} />
                  </div>
                  <div className="text-left">
                    <div className="text-lg font-semibold">Search Documents</div>
                    <div className="text-sm text-blue-100">Find and access medical data</div>
                  </div>
                </div>
                <div className="absolute inset-0 bg-white/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
              </button>

              <button
                onClick={handleUpload}
                className="group relative flex items-center justify-center gap-4 px-8 py-6 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-2xl hover:from-green-700 hover:to-emerald-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
                    <Upload size={24} />
                  </div>
                  <div className="text-left">
                    <div className="text-lg font-semibold">Upload Document</div>
                    <div className="text-sm text-green-100">Secure blockchain storage</div>
                  </div>
                </div>
                <div className="absolute inset-0 bg-white/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
              </button>
            </div>

            {!isWalletConnected && (
              <div className="mt-8 p-6 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-2xl">
                <div className="flex items-center justify-center gap-3">
                  <div className="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center">
                    <Wallet size={16} className="text-amber-600" />
                  </div>
                  <p className="text-amber-700 font-medium">
                    Connect your wallet to upload and manage documents
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Footer Stats */}
          <div className="mt-12 text-center">
            <div className="inline-flex items-center gap-6 px-6 py-3 bg-white/60 backdrop-blur-sm rounded-full border border-gray-200">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600">Blockchain Active</span>
              </div>
              <div className="w-px h-4 bg-gray-300"></div>
              <div className="flex items-center gap-2">
                <Globe size={14} className="text-gray-500" />
                <span className="text-sm text-gray-600">IPFS Network</span>
              </div>
            </div>
          </div>

        </div>
      </div>

      {isDropdownOpen && (
        <div 
          className="fixed inset-0 z-0" 
          onClick={() => setIsDropdownOpen(false)}
        />
      )}
    </div>
  );
}
