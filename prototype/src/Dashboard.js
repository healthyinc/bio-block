import React, { useState, useEffect, useCallback } from "react";
import { Wallet, FileText, Copy, ExternalLink, ArrowUpRight } from "lucide-react";
import {
  getEarnings,
  withdrawEarnings,
  getMyDocuments,
  getDocumentPrice,
  getWalletBalance,
} from "./contractService";
import { decryptFile } from "./encryptionUtils";

export default function Dashboard({ onBack, isWalletConnected, walletAddress }) {
  const [earnings, setEarnings] = useState("0");
  const [walletBalance, setWalletBalance] = useState("0");
  const [, setIsLoading] = useState(false);
  const [isWithdrawing, setIsWithdrawing] = useState(false);
  const [withdrawTx, setWithdrawTx] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [showDocuments, setShowDocuments] = useState(false);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [downloadingDocs, setDownloadingDocs] = useState({});

  const loadDashboardData = useCallback(async () => {
    if (!isWalletConnected) return;

    setIsLoading(true);
    try {
      const [userEarnings, balance] = await Promise.all([
        getEarnings(walletAddress),
        getWalletBalance(walletAddress),
      ]);
      setEarnings(userEarnings);
      setWalletBalance(balance);
    } catch (error) {
      console.error("Error loading dashboard:", error);
    } finally {
      setIsLoading(false);
    }
  }, [isWalletConnected, walletAddress]);

  const handleWithdraw = async () => {
    if (parseFloat(earnings) <= 0) {
      alert("No earnings to withdraw");
      return;
    }

    const confirmed = window.confirm(`Withdraw ${earnings} ETH to your wallet?`);
    if (!confirmed) return;

    setIsWithdrawing(true);
    setWithdrawTx(null);
    try {
      const txHash = await withdrawEarnings();
      setWithdrawTx(txHash);
      setEarnings("0");
      // Refresh wallet balance after withdrawal
      const newBalance = await getWalletBalance(walletAddress);
      setWalletBalance(newBalance);
    } catch (error) {
      console.error("Withdrawal error:", error);
      alert(`Withdrawal failed: ${error.message}`);
    } finally {
      setIsWithdrawing(false);
    }
  };

  const copyAddress = () => {
    navigator.clipboard.writeText(walletAddress);
    alert("Address copied to clipboard!");
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
            console.error("Error getting price for document:", hash, error);
            return { hash, price: "Error" };
          }
        })
      );
      setDocuments(documentsWithPrices);
      setShowDocuments(true);
    } catch (error) {
      console.error("Error loading documents:", error);
      alert("Failed to load documents");
    } finally {
      setLoadingDocuments(false);
    }
  };

  const handleDownload = async (doc, index) => {
    setDownloadingDocs((prev) => ({ ...prev, [index]: true }));

    try {
      const response = await fetch(`https://gateway.pinata.cloud/ipfs/${doc.hash}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch document: ${response.statusText}`);
      }

      const encryptedData = await response.text();
      const decryptedData = decryptFile(encryptedData);
      const bytes = new Uint8Array(
        atob(decryptedData)
          .split("")
          .map((char) => char.charCodeAt(0))
      );

      const blob = new Blob([bytes], {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = doc.hash;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download error:", error);
      alert(`Download failed: ${error.message}`);
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
          <p className="text-gray-600">Please connect your wallet to view your dashboard</p>
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
        {/* Stats & Actions Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
          <div className="flex items-center justify-between mb-6 pb-6 border-b border-gray-100">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center shadow-sm">
                <Wallet className="text-blue-600" size={24} />
              </div>
              <div>
                <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider">
                  Wallet Balance
                </p>
                <p className="text-3xl font-bold text-gray-900">
                  {parseFloat(walletBalance).toFixed(4)} ETH
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider">
                Platform Earnings
              </p>
              <p className="text-3xl font-bold text-green-600">{earnings} ETH</p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 mb-0">
            <button
              onClick={handleWithdraw}
              disabled={parseFloat(earnings) <= 0 || isWithdrawing}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-4 bg-green-600 text-white rounded-xl hover:bg-green-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-all font-bold text-lg shadow-sm hover:shadow-md"
            >
              {isWithdrawing ? "Processing..." : "Withdraw Earnings"}
              {!isWithdrawing && <ArrowUpRight size={20} />}
            </button>

            <button
              onClick={loadMyDocuments}
              disabled={loadingDocuments}
              className="flex-1 px-6 py-4 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-all font-bold text-lg shadow-sm hover:shadow-md text-center"
            >
              {loadingDocuments ? "Loading Documents..." : "My Documents"}
            </button>
          </div>

          {withdrawTx && (
            <div className="mt-6 p-4 bg-blue-50 border border-blue-100 rounded-xl flex items-center justify-between animate-in fade-in slide-in-from-top-4 duration-500">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <ExternalLink size={18} className="text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-blue-900 font-bold">Withdrawal Complete</p>
                  <p className="text-xs text-blue-700">Earnings have been sent to your wallet</p>
                </div>
              </div>
              <a
                href={`https://sepolia.etherscan.io/tx/${withdrawTx}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-white border border-blue-200 text-blue-700 hover:bg-blue-50 rounded-lg text-sm font-bold transition-colors flex items-center gap-2"
              >
                View on Etherscan
                <ExternalLink size={14} />
              </a>
            </div>
          )}
        </div>

        {/* Wallet Address Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gray-50 rounded-lg flex items-center justify-center border border-gray-100">
                <Wallet size={20} className="text-gray-500" />
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-widest font-black">
                  Connected Address
                </p>
                <p className="text-sm font-mono text-gray-600 tracking-tight">{walletAddress}</p>
              </div>
            </div>
            <button
              onClick={copyAddress}
              className="flex items-center gap-2 px-3 py-2 hover:bg-gray-50 rounded-lg transition-all text-gray-500 hover:text-indigo-600 font-bold text-xs border border-transparent hover:border-indigo-100"
              title="Copy Address"
            >
              <Copy size={16} />
              COPY
            </button>
          </div>
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
                        <h3 className="font-medium text-gray-800 mb-1">Document #{index + 1}</h3>
                        <p className="text-sm text-gray-600 break-all font-mono">{doc.hash}</p>
                      </div>
                      <div className="text-right ml-4">
                        <p className="text-lg font-bold text-green-600">{doc.price} ETH</p>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <button
                        onClick={() => handleDownload(doc, index)}
                        disabled={downloadingDocs[index]}
                        className="text-blue-600 hover:text-blue-800 text-sm disabled:text-gray-400 disabled:cursor-not-allowed"
                      >
                        {downloadingDocs[index] ? "Downloading..." : "Download"}
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
