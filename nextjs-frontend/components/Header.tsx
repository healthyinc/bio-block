"use client";

import { User, ChevronDown, Wallet } from "lucide-react";
import { useState } from "react";

interface HeaderProps {
  isWalletConnected: boolean;
  walletAddress: string;
  onConnect: () => void;
  onDisconnect: () => void;
  onDashboard?: () => void;
}

export default function Header({ 
  isWalletConnected, 
  walletAddress, 
  onConnect, 
  onDisconnect,
  onDashboard 
}: HeaderProps) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  return (
    <div className="absolute top-6 right-6 z-50">
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
              {onDashboard && (
                <button
                  onClick={() => {
                    onDashboard();
                    setIsDropdownOpen(false);
                  }}
                  className="w-full text-left px-5 py-4 text-sm text-gray-700 hover:bg-blue-50 transition-colors border-b border-gray-100 rounded-t-xl font-medium"
                >
                  View My Profile
                </button>
              )}
              <button
                onClick={() => {
                  onDisconnect();
                  setIsDropdownOpen(false);
                }}
                className="w-full text-left px-5 py-4 text-sm text-red-600 hover:bg-red-50 transition-colors rounded-b-xl font-medium"
              >
                Disconnect Wallet
              </button>
            </div>
          )}
        </div>
      ) : (
        <button
          onClick={onConnect}
          className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
        >
          <Wallet size={18} />
          <span className="font-medium">Connect Wallet</span>
        </button>
      )}
    </div>
  );
}
