"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Wallet, DollarSign, FileText } from 'lucide-react';
import {
  getEarnings,
  withdrawEarnings,
  getMyDocuments,
  getDocumentPrice,
} from '../lib/contractService';
import { decryptFile } from '../lib/encryptionUtils';

interface DashboardProps {
  onBack: () => void;
  isWalletConnected: boolean;
  walletAddress: string;
}

interface DocumentInfo {
  hash: string;
  price: string;
}

interface DownloadingState {
  [key: number]: boolean;
}

export default function Dashboard({
  onBack,
  isWalletConnected,
  walletAddress,
}: DashboardProps) {
  const [earnings, setEarnings] = useState<string>('0');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isWithdrawing, setIsWithdrawing] = useState<boolean>(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [showDocuments, setShowDocuments] = useState<boolean>(false);
  const [loadingDocuments, setLoadingDocuments] = useState<boolean>(false);
  const [downloadingDocs, setDownloadingDocs] = useState<DownloadingState>({});

  const loadDashboardData = useCallback(async () => {
    if (!isWalletConnected) return;

    setIsLoading(true);
    try {
      const userEarnings = await getEarnings(walletAddress);
      setEarnings(userEarnings);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setIsLoading(false);
    }
  }, [isWalletConnected, walletAddress]);

  const handleWithdraw = async () => {
    if (parseFloat(earnings) <= 0) {
      alert('No earnings to withdraw');
      return;
    }

    const confirmed = window.confirm(`Withdraw ${earnings} ETH to your wallet?`);
    if (!confirmed) return;

    setIsWithdrawing(true);
    try {
      const txHash = await withdrawEarnings();
      alert(`Withdrawal successful! Transaction: ${txHash}`);
      setEarnings('0');
    } catch (error) {
      console.error('Withdrawal error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Withdrawal failed: ${errorMessage}`);
    } finally {
      setIsWithdrawing(false);
    }
  };

  const loadMyDocuments = async () => {
    if (!isWalletConnected) return;

    setLoadingDocuments(true);
    try {
      const docHashes = await getMyDocuments();
      const documentsWithPrices = await Promise.all(
        docHashes.map(async (hash) => {
          try {
            const price = await getDocumentPrice(hash);
            return { hash, price };
          } catch (error) {
            console.error('Error getting price for document:', hash, error);
            return { hash, price: 'Error' };
          }
        })
      );
      setDocuments(documentsWithPrices);
      setShowDocuments(true);
    } catch (error) {
      console.error('Error loading documents:', error);
      alert('Failed to load documents');
    } finally {
      setLoadingDocuments(false);
    }
  };

  const handleDownload = async (doc: DocumentInfo, index: number) => {
    setDownloadingDocs((prev) => ({ ...prev, [index]: true }));

    try {
      const response = await fetch(
        `https://gateway.pinata.cloud/ipfs/${doc.hash}`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch document: ${response.statusText}`);
      }

      const encryptedData = await response.text();
      const decryptedData = decryptFile(encryptedData);
      const bytes = new Uint8Array(
        atob(decryptedData)
          .split('')
          .map((char) => char.charCodeAt(0))
      );

      const blob = new Blob([bytes], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = doc.hash;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Download failed: ${errorMessage}`);
    } finally {
      setDownloadingDocs((prev) => ({ ...prev, [index]: false }));
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, [isWalletConnected, walletAddress, loadDashboardData]);

  if (!isWalletConnected) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <Wallet size={48} className="mx-auto mb-4 text-gray-400" />
          <h2 className="text-xl font-bold mb-2">Connect Your Wallet</h2>
          <p className="text-gray-600">
            Please connect your wallet to view your dashboard
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="flex justify-between items-center mb-6">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          ← Back to Main
        </button>
        <h1 className="text-2xl font-bold text-gray-800">My Dashboard</h1>
        <div></div>
      </div>

      <div className="max-w-6xl mx-auto space-y-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <DollarSign size={24} className="text-green-600" />
              <h2 className="text-xl font-semibold">Earnings</h2>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-green-600">{earnings} ETH</p>
              <p className="text-sm text-gray-500">Available to withdraw</p>
            </div>
          </div>

          <button
            onClick={handleWithdraw}
            disabled={parseFloat(earnings) <= 0 || isWithdrawing}
            className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors mb-3"
          >
            {isWithdrawing ? 'Withdrawing...' : 'Withdraw Earnings'}
          </button>

          <button
            onClick={loadMyDocuments}
            disabled={loadingDocuments}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {loadingDocuments ? 'Loading...' : 'My Documents'}
          </button>
        </div>

        {showDocuments && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <FileText size={24} className="text-blue-600" />
                <h2 className="text-xl font-semibold">My Documents</h2>
              </div>
              <button
                onClick={() => setShowDocuments(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            {documents.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No documents found</p>
            ) : (
              <div className="space-y-4">
                {documents.map((doc, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-800 mb-1">
                          Document #{index + 1}
                        </h3>
                        <p className="text-sm text-gray-600 break-all font-mono">
                          {doc.hash}
                        </p>
                      </div>
                      <div className="text-right ml-4">
                        <p className="text-lg font-bold text-green-600">
                          {doc.price} ETH
                        </p>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <button
                        onClick={() => handleDownload(doc, index)}
                        disabled={downloadingDocs[index]}
                        className="text-blue-600 hover:text-blue-800 text-sm disabled:text-gray-400 disabled:cursor-not-allowed"
                      >
                        {downloadingDocs[index] ? 'Downloading...' : 'Download'}
                      </button>
                      <button
                        onClick={() => navigator.clipboard.writeText(doc.hash)}
                        className="text-gray-500 hover:text-gray-700 text-sm"
                      >
                        Copy Hash
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
