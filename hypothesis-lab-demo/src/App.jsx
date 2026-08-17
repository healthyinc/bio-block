import { useState, useCallback, useEffect } from 'react';
import './App.css';

import UploadScreen from './components/UploadScreen';
import DatasetPane from './components/DatasetPane';
import TreePane from './components/TreePane';
import ResearchPane from './components/ResearchPane';
import ExportModal from './components/ExportModal';
import BranchDiffModal from './components/BranchDiffModal';
import ProvenanceLedger from './components/ProvenanceLedger';

import {
  createSession,
  getSession,
  submitAnswer,
  forkBranch,
  runAnalysis,
  deleteSession,
} from './api/client';

import { connectWallet, truncateAddress } from './web3/wallet';

function getInitialTheme() {
  if (typeof window === 'undefined') return 'dark';
  return document.documentElement.getAttribute('data-theme') || 'dark';
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [profile, setProfile] = useState(null);
  const [tree, setTree] = useState(null);
  const [activeBranchId, setActiveBranchId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [hypotheses, setHypotheses] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [validation, setValidation] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [answering, setAnswering] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);

  const [showExportModal, setShowExportModal] = useState(false);
  const [showDiffModal, setShowDiffModal] = useState(false);
  const [showLedger, setShowLedger] = useState(false);

  const [walletAddress, setWalletAddress] = useState(null);
  const [walletProvider, setWalletProvider] = useState(null);
  const [walletSigner, setWalletSigner] = useState(null);
  const [walletConnecting, setWalletConnecting] = useState(false);

  // theme
  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('hypothesis-lab-theme', next);
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // wallet
  const handleConnectWallet = useCallback(async () => {
    setWalletConnecting(true);
    try {
      const { address, provider, signer } = await connectWallet();
      setWalletAddress(address);
      setWalletProvider(provider);
      setWalletSigner(signer);
    } catch (err) {
      setError(err.message);
    } finally {
      setWalletConnecting(false);
    }
  }, []);

  const handleDisconnectWallet = useCallback(() => {
    setWalletAddress(null);
    setWalletProvider(null);
    setWalletSigner(null);
  }, []);

  // handle metamask account / chain changes
  useEffect(() => {
    if (!window.ethereum) return;

    const handleAccountsChanged = (accounts) => {
      if (accounts.length === 0) {
        handleDisconnectWallet();
      } else if (accounts[0].toLowerCase() !== walletAddress?.toLowerCase()) {
        handleConnectWallet();
      }
    };

    const handleChainChanged = () => {
      if (walletAddress) handleConnectWallet();
    };

    window.ethereum.on('accountsChanged', handleAccountsChanged);
    window.ethereum.on('chainChanged', handleChainChanged);

    return () => {
      window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
      window.ethereum.removeListener('chainChanged', handleChainChanged);
    };
  }, [walletAddress, handleConnectWallet, handleDisconnectWallet]);

  const handleUpload = useCallback(async (file) => {
    const data = await createSession(file);
    setSessionId(data.session_id);
    setProfile(data.profile);
    setTree(data.tree);
    setActiveBranchId(data.tree.active_branch_id);
    setCurrentQuestion(data.current_question);
    setHypotheses(data.candidate_hypotheses || []);
    setAnalyses(data.candidate_analyses || []);
    setValidation(data.validation);
    setAnalysisResult(null);
    setError(null);
  }, []);

  const handleAnswer = useCallback(async ({ optionId, customAnswer }) => {
    if (!sessionId || !tree) return;
    setAnswering(true);
    setError(null);
    try {
      const branch = tree.branches[activeBranchId];
      const parentNodeId = branch.node_ids[branch.node_ids.length - 1];

      const data = await submitAnswer(sessionId, {
        parentNodeId,
        optionId,
        customAnswer,
      });

      setTree(data.tree);
      if (data.tree.active_branch_id) {
        setActiveBranchId(data.tree.active_branch_id);
      }
      setCurrentQuestion(data.current_question);
      setHypotheses(data.candidate_hypotheses || []);
      setAnalyses(data.candidate_analyses || []);
      setValidation(data.validation);
      setAnalysisResult(data.analysis_result || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnswering(false);
    }
  }, [sessionId, tree, activeBranchId]);

  const handleFork = useCallback(async (nodeId) => {
    if (!sessionId) return;
    setError(null);
    try {
      const data = await forkBranch(sessionId, nodeId);
      setTree(data.tree);
      setActiveBranchId(data.new_branch_id);
      setCurrentQuestion(data.current_question);
      setHypotheses([]);
      setAnalyses([]);
      setAnalysisResult(null);
    } catch (err) {
      setError(err.message);
    }
  }, [sessionId]);

  const handleSwitchBranch = useCallback(async (branchId) => {
    if (!sessionId) return;
    setActiveBranchId(branchId);
    try {
      const data = await getSession(sessionId, branchId);
      setTree(data.tree);
      setCurrentQuestion(data.current_question);
      setHypotheses(data.candidate_hypotheses || []);
      setAnalyses(data.candidate_analyses || []);
      setValidation(data.validation);
      setAnalysisResult(data.analysis_result || null);
    } catch (err) {
      setAnalysisResult(null);
      setError(err.message);
    }
  }, [sessionId]);

  const handleRunAnalysis = useCallback(async (hypothesisId, analysisId) => {
    if (!sessionId) return;
    setAnswering(true);
    setError(null);
    try {
      const data = await runAnalysis(sessionId, { hypothesisId, analysisId });
      setTree(data.tree);
      setAnalysisResult(data.result);
      setCurrentQuestion(data.follow_up_question);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnswering(false);
    }
  }, [sessionId]);

  const handleReset = useCallback(async () => {
    if (sessionId) {
      try { await deleteSession(sessionId); } catch { /* ignore */ }
    }
    setSessionId(null);
    setProfile(null);
    setTree(null);
    setActiveBranchId(null);
    setCurrentQuestion(null);
    setHypotheses([]);
    setAnalyses([]);
    setValidation(null);
    setAnalysisResult(null);
    setError(null);
  }, [sessionId]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-logo">
            <span className="app-logo-mark">BIO-BLOCK</span>
            <span className="app-logo-title">ANALYTICS LAB</span>
          </div>
          <span className="prototype-badge">On-Chain Provenance v1.0</span>
        </div>
        <div className="app-header-actions">
          {sessionId && (
            <>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowExportModal(true)}>
                EXPORT REPORT
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowDiffModal(true)}>
                COMPARE BRANCHES
              </button>
            </>
          )}

          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setShowLedger(true)}
            title="View on-chain analysis history"
          >
            ⛓ PROVENANCE LEDGER
          </button>

          {walletAddress ? (
            <button
              className="wallet-btn wallet-btn--connected"
              onClick={handleDisconnectWallet}
              title={`Connected: ${walletAddress}\nClick to disconnect`}
            >
              <span className="wallet-dot" />
              <span className="wallet-addr">{truncateAddress(walletAddress)}</span>
            </button>
          ) : (
            <button
              className="wallet-btn wallet-btn--disconnected"
              onClick={handleConnectWallet}
              disabled={walletConnecting}
            >
              {walletConnecting ? 'CONNECTING…' : '🦊 CONNECT WALLET'}
            </button>
          )}

          <button className="theme-toggle" onClick={toggleTheme}>
            <span>{theme === 'dark' ? 'LIGHT MODE' : 'DARK MODE'}</span>
          </button>
          {sessionId && (
            <button className="btn btn-danger btn-sm" onClick={handleReset}>
              CLEAR SESSION
            </button>
          )}
        </div>
      </header>

      {error && (
        <div style={{ margin: '8px 16px', padding: '8px 12px', background: 'var(--accent-red-bg)', border: '1px solid var(--accent-red)', color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
          ERROR: {error}
          <button
            className="btn btn-ghost btn-sm"
            style={{ marginLeft: 'auto', float: 'right' }}
            onClick={() => setError(null)}
          >
            DISMISS
          </button>
        </div>
      )}

      {!sessionId ? (
        <UploadScreen onUpload={handleUpload} />
      ) : (
        <div className="workspace">
          <DatasetPane profile={profile} />
          <TreePane
            tree={tree}
            activeBranchId={activeBranchId}
            onSwitchBranch={handleSwitchBranch}
            onFork={handleFork}
            onOpenDiff={() => setShowDiffModal(true)}
            onOpenExport={() => setShowExportModal(true)}
          />
          <ResearchPane
            profile={profile}
            currentQuestion={currentQuestion}
            hypotheses={hypotheses}
            analyses={analyses}
            validation={validation}
            analysisResult={analysisResult}
            onAnswer={handleAnswer}
            onRunAnalysis={handleRunAnalysis}
            answering={answering}
          />
        </div>
      )}

      <ExportModal
        tree={tree}
        activeBranchId={activeBranchId}
        profile={profile}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        walletAddress={walletAddress}
        signer={walletSigner}
        sessionId={sessionId}
      />

      <BranchDiffModal
        tree={tree}
        activeBranchId={activeBranchId}
        isOpen={showDiffModal}
        onClose={() => setShowDiffModal(false)}
        onSwitchBranch={handleSwitchBranch}
      />

      <ProvenanceLedger
        isOpen={showLedger}
        onClose={() => setShowLedger(false)}
        walletAddress={walletAddress}
        provider={walletProvider}
      />
    </div>
  );
}
