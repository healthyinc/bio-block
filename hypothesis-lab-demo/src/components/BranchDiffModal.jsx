import React, { useState, useEffect } from 'react';

export default function BranchDiffModal({ tree, activeBranchId, isOpen, onClose, onSwitchBranch }) {
  const branches = Object.values(tree?.branches || {});

  const primaryBranch = branches.find(b => b.is_primary) || branches[0];
  const activeBranch = tree?.branches?.[activeBranchId] || branches.find(b => b.id === activeBranchId) || branches[0];

  const defaultAId = primaryBranch ? primaryBranch.id : branches[0]?.id;
  const defaultBId = (activeBranch && activeBranch.id !== defaultAId)
    ? activeBranch.id
    : (branches.find(b => b.id !== defaultAId)?.id || defaultAId);

  const [branchAId, setBranchAId] = useState(defaultAId);
  const [branchBId, setBranchBId] = useState(defaultBId);

  useEffect(() => {
    if (defaultAId) setBranchAId(defaultAId);
    if (defaultBId) setBranchBId(defaultBId);
  }, [isOpen, activeBranchId, defaultAId, defaultBId]);

  if (!isOpen || !tree || !tree.branches) return null;

  if (branches.length < 2) {
    return (
      <div style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 99999
      }}>
        <div style={{ background: 'var(--bg-secondary)', padding: '24px', border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-card)', maxWidth: '440px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', fontSize: '13px', fontWeight: 700, marginBottom: '10px' }}>
            BRANCH COMPARISON REQUIRES 2+ BRANCHES
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: '1.5' }}>
            Fork an intermediate decision node in your exploration tree to compare statistical outcomes across alternative analytical paths.
          </p>
          <button className="btn btn-primary btn-sm" onClick={onClose}>CLOSE</button>
        </div>
      </div>
    );
  }

  const branchA = tree.branches[branchAId] || branches[0];
  const branchB = tree.branches[branchBId] || branches[1] || branches[0];

  function getBranchResult(branch) {
    if (!branch || !branch.node_ids || !tree.nodes) return null;

    // find latest result node
    for (let i = branch.node_ids.length - 1; i >= 0; i--) {
      const node = tree.nodes[branch.node_ids[i]];
      if (!node) continue;

      const isResultKind = node.kind === 'result' || node.kind === 'RESULT';
      const hasResultCtx = Boolean(node.context?.result || node.context?.test_used || node.context?.p_value);

      if (isResultKind || hasResultCtx) {
        const testUsed = node.context?.test_used || node.context?.test_name || 'Statistical Analysis';
        const resObj = node.context?.result || node.context || {};
        const pValue = resObj.p_value ?? node.context?.p_value ?? null;
        const statistic = resObj.statistic ?? node.context?.statistic ?? null;
        const significant = resObj.significant ?? (pValue != null ? pValue < 0.05 : false);
        const effectSize = node.context?.effect_size || resObj.effect_size || null;

        return {
          nodeId: node.id,
          testUsed,
          resObj,
          pValue,
          statistic,
          significant,
          effectSize,
          rawNode: node
        };
      }
    }
    return null;
  }

  const resA = getBranchResult(branchA);
  const resB = getBranchResult(branchB);

  function renderBranchCard(branch, res, currentBranchId, onSelectBranchId) {
    const isCurrentActive = branch.id === activeBranchId;

    return (
      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-default)', padding: '16px', borderRadius: 'var(--radius-card)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', fontWeight: 700, color: 'var(--accent-amber)', letterSpacing: '0.5px' }}>
              {branch.is_primary ? 'MAIN EXPLORATION' : 'FORKED BRANCH'} {isCurrentActive ? '(ACTIVE)' : ''}
            </span>
            <button
              className="btn btn-ghost btn-sm"
              style={{ fontSize: '10px', padding: '3px 8px' }}
              onClick={() => { onSwitchBranch(branch.id); onClose(); }}
            >
              SWITCH TO THIS
            </button>
          </div>

          <select
            value={currentBranchId}
            onChange={(e) => onSelectBranchId(e.target.value)}
            style={{ width: '100%', marginBottom: '6px' }}
          >
            {branches.map(b => (
              <option key={b.id} value={b.id}>
                {b.name || b.id} {b.is_primary ? '(Main)' : ''} {b.id === activeBranchId ? '[Active]' : ''}
              </option>
            ))}
          </select>

          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
            Depth: {branch.node_ids?.length || 0} decision steps
          </div>
        </div>

        {res ? (
          <div style={{
            padding: '12px',
            background: 'var(--bg-panel)',
            borderLeft: `4px solid ${res.significant ? 'var(--accent-green)' : 'var(--accent-amber)'}`,
            borderRadius: 'var(--radius-sharp)'
          }}>
            <div style={{ fontWeight: 700, fontSize: '12px', color: 'var(--text-primary)', marginBottom: '6px' }}>
              {res.testUsed}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                fontWeight: 700,
                color: res.significant ? 'var(--accent-green)' : 'var(--accent-amber)'
              }}>
                p = {res.pValue != null ? Number(res.pValue).toFixed(6) : 'N/A'}
              </span>
              <span style={{
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                padding: '2px 6px',
                borderRadius: '2px',
                background: res.significant ? 'var(--accent-green-bg)' : 'var(--accent-amber-bg)',
                color: res.significant ? 'var(--accent-green)' : 'var(--accent-amber)',
                border: `1px solid ${res.significant ? 'var(--accent-green)' : 'var(--accent-amber-border)'}`
              }}>
                {res.significant ? 'Significant (p < 0.05)' : 'Non-Significant'}
              </span>
            </div>

            {res.statistic != null && (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Test Statistic: <strong>{Number(res.statistic).toFixed(4)}</strong>
              </div>
            )}

            {res.effectSize && (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', marginTop: '6px', paddingTop: '6px', borderTop: '1px stroke var(--border-subtle)' }}>
                Effect Size: {res.effectSize.metric || 'd'} = {res.effectSize.value ?? 'N/A'} {res.effectSize.magnitude ? `(${res.effectSize.magnitude})` : ''}
              </div>
            )}
          </div>
        ) : (
          <div style={{
            padding: '16px',
            background: 'var(--bg-panel)',
            border: '1px dashed var(--border-subtle)',
            borderRadius: 'var(--radius-sharp)',
            textAlign: 'center',
            color: 'var(--text-muted)',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)'
          }}>
            No statistical result executed yet on this branch.
            <div style={{ marginTop: '8px' }}>
              <button
                className="btn btn-ghost btn-sm"
                style={{ fontSize: '10px' }}
                onClick={() => { onSwitchBranch(branch.id); onClose(); }}
              >
                Switch & Run Analysis →
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  // calculate delta if both have results
  let deltaText = null;
  if (resA && resB && resA.pValue != null && resB.pValue != null) {
    const diff = Math.abs(resA.pValue - resB.pValue);
    if (resA.significant === resB.significant) {
      deltaText = `Both branches agree on significance decision (${resA.significant ? 'Reject H₀' : 'Fail to reject H₀'}), with Δp = ${diff.toFixed(6)}.`;
    } else {
      deltaText = `Branch outcome disagreement: ${branchA.name || 'Branch 1'} is ${resA.significant ? 'Significant' : 'Non-Significant'}, whereas ${branchB.name || 'Branch 2'} is ${resB.significant ? 'Significant' : 'Non-Significant'} (Δp = ${diff.toFixed(6)}).`;
    }
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 99999, padding: '20px'
    }}>
      <div style={{
        maxWidth: '860px', width: '100%', background: 'var(--bg-secondary)',
        border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-card)',
        boxShadow: 'var(--shadow-md)', overflow: 'hidden'
      }}>
        <div style={{ padding: '14px 18px', background: 'var(--bg-panel)', borderBottom: '1px solid var(--border-default)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 700, color: 'var(--accent-amber)', letterSpacing: '0.5px' }}>
              SIDE-BY-SIDE BRANCH COMPARISON & STATISTICAL DIFF
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
              Select any two branches to evaluate robustness and methodological sensitivity side by side.
            </div>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose} style={{ fontSize: '16px', lineHeight: '1' }}>×</button>
        </div>

        <div style={{ padding: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {renderBranchCard(branchA, resA, branchAId, setBranchAId)}
          {renderBranchCard(branchB, resB, branchBId, setBranchBId)}
        </div>

        {deltaText && (
          <div style={{ margin: '0 20px 20px 20px', padding: '12px 14px', background: 'var(--accent-amber-bg)', border: '1px solid var(--accent-amber-border)', borderRadius: 'var(--radius-sharp)', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--accent-amber)' }}>
            <strong>STATISTICAL DIFF:</strong> {deltaText}
          </div>
        )}

        <div style={{ padding: '12px 18px', background: 'var(--bg-panel)', borderTop: '1px solid var(--border-default)', textAlign: 'right' }}>
          <button className="btn btn-primary btn-sm" onClick={onClose}>CLOSE COMPARISON</button>
        </div>
      </div>
    </div>
  );
}
