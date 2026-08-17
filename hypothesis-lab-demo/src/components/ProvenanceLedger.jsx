import { useState, useEffect, useCallback } from 'react';
import { getAnalyticsCount, getAnalyticsForAddress, truncateAddress } from '../web3/wallet';
import { SEPOLIA_ETHERSCAN, PINATA_GATEWAY } from '../web3/constants';

const PAGE_SIZE = 10;

export default function ProvenanceLedger({ isOpen, onClose, walletAddress, provider }) {
  const [records, setRecords] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRecords = useCallback(async (newOffset = 0) => {
    if (!provider || !walletAddress) return;
    setLoading(true);
    setError(null);
    try {
      const count = await getAnalyticsCount(provider, walletAddress);
      setTotalCount(count);
      if (count === 0) {
        setRecords([]);
        return;
      }
      const data = await getAnalyticsForAddress(provider, walletAddress, newOffset, PAGE_SIZE);
      setRecords(data);
      setOffset(newOffset);
    } catch (err) {
      setError(err.message || 'Failed to fetch on-chain records.');
    } finally {
      setLoading(false);
    }
  }, [provider, walletAddress]);

  useEffect(() => {
    if (isOpen && walletAddress && provider) {
      fetchRecords(0);
    }
  }, [isOpen, walletAddress, provider, fetchRecords]);

  if (!isOpen) return null;

  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < totalCount;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 99999,
      padding: '20px',
    }}>
      <div style={{
        maxWidth: '900px',
        width: '100%',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-strong)',
        borderRadius: 'var(--radius-card)',
        boxShadow: 'var(--shadow-md)',
        display: 'flex',
        flexDirection: 'column',
        maxHeight: '85vh',
        overflow: 'hidden',
      }}>
        <div style={{
          padding: '12px 18px',
          background: 'var(--bg-panel)',
          borderBottom: '1px solid var(--border-default)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '13px',
            fontWeight: 700,
            color: 'var(--accent-amber)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '24px',
              height: '24px',
              background: 'var(--accent-amber-bg)',
              border: '1px solid var(--accent-amber)',
              borderRadius: '4px',
              fontSize: '12px',
            }}>⛓</span>
            PROVENANCE LEDGER
            {totalCount > 0 && (
              <span style={{
                fontSize: '11px',
                color: 'var(--text-tertiary)',
                fontWeight: 400,
              }}>
                ({totalCount} record{totalCount !== 1 ? 's' : ''})
              </span>
            )}
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>×</button>
        </div>

        <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
          {!walletAddress ? (
            <div style={{
              textAlign: 'center',
              padding: '40px 20px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-tertiary)',
              fontSize: '13px',
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px', opacity: 0.4 }}>🔒</div>
              Connect your wallet to view on-chain analysis history.
            </div>
          ) : loading ? (
            <div style={{
              textAlign: 'center',
              padding: '40px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)',
              fontSize: '13px',
            }}>
              <div className="spinner" style={{ margin: '0 auto 12px' }} />
              QUERYING SEPOLIA CHAIN...
            </div>
          ) : error ? (
            <div style={{
              padding: '12px',
              background: 'var(--accent-red-bg)',
              border: '1px solid var(--accent-red)',
              color: 'var(--accent-red)',
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
            }}>
              ERROR: {error}
            </div>
          ) : records.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '40px 20px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-tertiary)',
              fontSize: '13px',
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px', opacity: 0.4 }}>📭</div>
              No on-chain attestations found for this wallet.
              <br />
              Complete an analysis and click "Save & Attest On-Chain" to create your first record.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
              }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-strong)' }}>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>SOURCE DATASET</th>
                    <th style={thStyle}>RESULT CID</th>
                    <th style={thStyle}>ANALYSIS TYPE</th>
                    <th style={thStyle}>TIMESTAMP</th>
                    <th style={thStyle}>IPFS</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((rec, idx) => (
                    <tr key={idx} style={{
                      borderBottom: '1px solid var(--border-subtle)',
                      transition: 'background 0.15s',
                    }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      <td style={tdStyle}>{offset + idx + 1}</td>
                      <td style={tdStyle} title={rec.sourceCID}>
                        {rec.sourceCID.length > 20
                          ? `${rec.sourceCID.slice(0, 10)}…${rec.sourceCID.slice(-6)}`
                          : rec.sourceCID
                        }
                      </td>
                      <td style={tdStyle} title={rec.resultCID}>
                        <span style={{ color: 'var(--accent-teal)' }}>
                          {rec.resultCID.slice(0, 10)}…{rec.resultCID.slice(-6)}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        <span style={{
                          padding: '2px 6px',
                          background: 'var(--accent-amber-bg)',
                          border: '1px solid var(--accent-amber)',
                          borderRadius: '3px',
                          fontSize: '10px',
                          textTransform: 'uppercase',
                        }}>
                          {rec.analysisType}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {new Date(rec.timestamp * 1000).toLocaleString()}
                      </td>
                      <td style={tdStyle}>
                        <a
                          href={`${PINATA_GATEWAY}/${rec.resultCID}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            color: 'var(--accent-teal)',
                            textDecoration: 'none',
                            fontSize: '10px',
                          }}
                        >
                          VIEW ↗
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div style={{
          padding: '10px 18px',
          borderTop: '1px solid var(--border-default)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'var(--bg-panel)',
        }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--text-tertiary)',
          }}>
            {walletAddress && (
              <>Wallet: {truncateAddress(walletAddress)} · Sepolia Testnet</>
            )}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {walletAddress && records.length > 0 && (
              <>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => fetchRecords(offset - PAGE_SIZE)}
                  disabled={!canPrev}
                >
                  ← PREV
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => fetchRecords(offset + PAGE_SIZE)}
                  disabled={!canNext}
                >
                  NEXT →
                </button>
              </>
            )}
            <button className="btn btn-ghost btn-sm" onClick={onClose}>
              CLOSE
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const thStyle = {
  textAlign: 'left',
  padding: '8px 10px',
  color: 'var(--text-tertiary)',
  fontWeight: 600,
  fontSize: '10px',
  letterSpacing: '0.05em',
  whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '8px 10px',
  color: 'var(--text-primary)',
  whiteSpace: 'nowrap',
};
