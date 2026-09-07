import { useState } from 'react';
import { useTooltip, TooltipPortal } from './Tooltip';

const FRIENDLY_OPTION_LABELS = {
  followup_covariate: 'Add a covariate & re-analyze',
  followup_nonparametric: 'Compare with non-parametric alternative',
  followup_subgroup: 'Test a subgroup or interaction',
  followup_new_hypothesis: 'Create related hypothesis with different outcome',
};

export default function TreePane({ tree, activeBranchId, onSwitchBranch, onFork, onOpenDiff, onOpenExport }) {
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [showPathSummary, setShowPathSummary] = useState(false);
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();

  if (!tree) return null;

  const branches = tree.branches ? Object.values(tree.branches) : [];
  const activeBranch = tree.branches?.[activeBranchId];
  const nodes = tree.nodes || {};

  function getNodeIcon(kind) {
    switch (kind) {
      case 'root': return '◉';
      case 'question': return '?';
      case 'answer': return '→';
      case 'hypothesis': return '◇';
      case 'result': return '■';
      default: return '·';
    }
  }

  function truncate(text, max = 36) {
    if (!text) return '';
    return text.length > max ? text.substring(0, max) + '…' : text;
  }

  const ancestorNodeIds = new Set();
  if (hoveredNodeId && activeBranch) {
    const idx = activeBranch.node_ids.indexOf(hoveredNodeId);
    if (idx !== -1) {
      for (let i = 0; i <= idx; i++) {
        ancestorNodeIds.add(activeBranch.node_ids[i]);
      }
    }
  }

  function getReadableOptionLabel(optIdOrRawText, optionsList = []) {
    if (!optIdOrRawText) return null;
    const foundObj = optionsList.find(
      (o) => o.id === optIdOrRawText || o.label === optIdOrRawText
    );
    if (foundObj && foundObj.label) return foundObj.label;
    if (FRIENDLY_OPTION_LABELS[optIdOrRawText]) return FRIENDLY_OPTION_LABELS[optIdOrRawText];
    return optIdOrRawText;
  }

  const pathSteps = [];
  if (activeBranch) {
    for (let i = 0; i < activeBranch.node_ids.length; i++) {
      const nid = activeBranch.node_ids[i];
      const node = nodes[nid];
      if (!node) continue;

      if (node.kind === 'answer') {
        const qNode = i > 0 ? nodes[activeBranch.node_ids[i - 1]] : null;
        const options = qNode?.options || [];
        const rawAnswer = node.answer || node.answer_option_id;
        const chosenLabel = getReadableOptionLabel(rawAnswer, options) || 'Selected Option';

        const unchosenObjs = options.filter(
          (o) => o.id !== node.answer_option_id && o.label !== node.answer && o.id !== node.answer
        );
        const unchosenLabels = unchosenObjs.map((u) => u.label || u.id);

        pathSteps.push({
          step: pathSteps.length + 1,
          question: qNode?.prompt || 'Decision Point',
          chosenLabel,
          unchosenLabels,
        });
      } else if (node.kind === 'result') {
        const resData = node.context?.result || {};
        pathSteps.push({
          step: pathSteps.length + 1,
          isResult: true,
          testUsed: node.context?.test_used || 'Analysis Result',
          significant: resData.significant || (resData.p_value < 0.05),
          pValue: resData.p_value,
        });
      }
    }
  }

  function renderNodeTooltipContent(node, nodeIndex) {
    if (node.kind === 'root') {
      return (
        <div className="tooltip-node-content">
          <div className="tooltip-node-title">◉ ROOT DATASET NODE</div>
          <div className="tooltip-node-desc">Starting point for dataset analysis & hypothesis generation.</div>
        </div>
      );
    }

    if (node.kind === 'result') {
      const resData = node.context?.result || {};
      const isSig = resData.significant || (resData.p_value < 0.05);
      return (
        <div className="tooltip-node-content">
          <div className="tooltip-node-title">TEST RESULT: {node.context?.test_used || 'Statistical Test'}</div>
          <div className="tooltip-node-desc">
            Status: <strong className={isSig ? 'text-sig' : 'text-nonsig'}>
              {isSig ? 'Statistically Significant (Reject H₀ at α = 0.05)' : 'Statistically Non-Significant (Fail to reject H₀ at α = 0.05)'}
            </strong> {resData.p_value != null && `(p = ${Number(resData.p_value).toFixed(6)})`}
          </div>
        </div>
      );
    }

    let qNode = null;
    let answerNode = null;

    if (node.kind === 'answer') {
      answerNode = node;
      qNode = nodes[node.parent_id];
    } else {
      qNode = node;
      if (activeBranch) {
        const idx = activeBranch.node_ids.indexOf(node.id);
        if (idx !== -1 && idx < activeBranch.node_ids.length - 1) {
          const childId = activeBranch.node_ids[idx + 1];
          if (nodes[childId] && nodes[childId].kind === 'answer') {
            answerNode = nodes[childId];
          }
        }
      }
    }

    const promptText = qNode?.prompt || (node.kind === 'answer' ? 'Research Question' : node.prompt || 'Decision Point');
    const options = qNode?.options || [];

    let chosenLabel = null;
    let unchosen = [];

    if (answerNode) {
      const rawAnswer = answerNode.answer || answerNode.answer_option_id;
      chosenLabel = getReadableOptionLabel(rawAnswer, options);
      unchosen = options.filter(
        (o) => o.id !== answerNode.answer_option_id && o.label !== answerNode.answer && o.id !== answerNode.answer
      );
    }

    return (
      <div className="tooltip-node-content">
        <div className="tooltip-node-step">DECISION STEP #{nodeIndex + 1}</div>
        <div className="tooltip-node-prompt">{promptText}</div>

        {chosenLabel ? (
          <div className="tooltip-chosen-box">
            <div className="tooltip-chosen-tag">CHOSEN PATH</div>
            <div className="tooltip-chosen-label">{chosenLabel}</div>
          </div>
        ) : (
          <div className="tooltip-chosen-box" style={{ background: 'var(--accent-amber-bg)', borderColor: 'var(--accent-amber-border)' }}>
            <div className="tooltip-chosen-tag" style={{ color: 'var(--accent-amber)' }}>PENDING DECISION</div>
            <div className="tooltip-chosen-label" style={{ color: 'var(--text-primary)', fontSize: 11 }}>Awaiting user selection from available options</div>
          </div>
        )}

        {unchosen.length > 0 && (
          <div className="tooltip-unchosen-box">
            <div className="tooltip-unchosen-tag">ALTERNATIVES NOT CHOSEN ({unchosen.length})</div>
            {unchosen.map((opt, idx) => (
              <div key={idx} className="tooltip-unchosen-item">
                <span className="bullet">·</span>
                <span>{getReadableOptionLabel(opt.label || opt.id, options)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="pane tree-pane">
      <div className="pane-header">
        <div style={{ flex: 1, minWidth: 0, marginRight: '8px' }}>
          <div className="pane-title" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            DECISION TREE
          </div>
          <div className="pane-subtitle" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            Branching path & audit
          </div>
        </div>
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center', flexShrink: 0 }}>
          <button
            className={`btn btn-xs ${showPathSummary ? 'btn-primary' : 'btn-ghost'}`}
            style={{ fontSize: '10px', padding: '2px 6px' }}
            onClick={() => setShowPathSummary(!showPathSummary)}
          >
            Audit Path
          </button>
          <button className="btn btn-ghost btn-xs" style={{ fontSize: '10px', padding: '2px 6px' }} onClick={onOpenExport}>
            Export
          </button>
          <button className="btn btn-ghost btn-xs" style={{ fontSize: '10px', padding: '2px 6px' }} onClick={onOpenDiff}>
            Compare
          </button>
        </div>
      </div>

      {branches.length > 1 && (
        <div style={{ padding: '10px 14px 0 14px' }}>
          <div className="branch-tabs">
            {branches.map((branch) => (
              <button
                key={branch.id}
                className={`branch-tab ${branch.id === activeBranchId ? 'active' : ''}`}
                onClick={() => onSwitchBranch(branch.id)}
              >
                {branch.name || 'Branch'} {branch.is_primary ? '(Main)' : ''}
              </button>
            ))}
          </div>
        </div>
      )}

      {showPathSummary && (
        <div style={{
          margin: '10px 14px',
          padding: '12px',
          background: 'var(--bg-panel)',
          border: '1px solid var(--accent-amber-border)',
          borderRadius: 'var(--radius-card)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontWeight: 700, fontSize: 12, color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
              DECISION PATH & ALTERNATIVES
            </span>
            <button className="btn btn-ghost btn-sm" style={{ padding: '0 4px' }} onClick={() => setShowPathSummary(false)}>×</button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '220px', overflowY: 'auto' }}>
            {pathSteps.map((s, idx) => (
              <div key={idx} style={{ padding: '6px 8px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 2 }}>
                <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-amber)', fontSize: 11 }}>{s.step}.</span>
                  {s.isResult ? (
                    <span style={{ fontWeight: 600, color: 'var(--accent-green)', fontSize: 11 }}>
                      {s.testUsed} → {s.significant ? 'Significant' : 'Not Significant'} (p={s.pValue?.toFixed(4) ?? 'N/A'})
                    </span>
                  ) : (
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 11 }}>
                      {s.question}
                    </span>
                  )}
                </div>

                {!s.isResult && (
                  <div style={{ marginLeft: 16, marginTop: 3 }}>
                    <div style={{ color: 'var(--accent-green)', fontSize: 11, fontWeight: 600 }}>
                      Chosen: {s.chosenLabel}
                    </div>
                    {s.unchosenLabels && s.unchosenLabels.length > 0 && (
                      <div style={{ marginTop: 2, color: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
                        <span>Unchosen: </span>
                        {s.unchosenLabels.join(' · ')}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="pane-body">
        {activeBranch ? (
          <ul className="tree-list">
            {activeBranch.node_ids.map((nodeId, idx) => {
              const node = nodes[nodeId];
              if (!node) return null;
              const isActive = idx === activeBranch.node_ids.length - 1;
              const isAncestor = ancestorNodeIds.has(nodeId);

              const fullLabel = node.kind === 'answer' && node.answer
                ? getReadableOptionLabel(node.answer, (nodes[node.parent_id]?.options || []))
                : node.kind === 'question' && node.prompt
                  ? node.prompt
                  : node.kind === 'result'
                    ? `Result: ${node.context?.test_used || 'Test'}`
                    : node.prompt || node.kind;

              return (
                <li key={nodeId} className="tree-item">
                  <div
                    className={`tree-node ${isActive ? 'active' : ''} ${isAncestor ? 'path-highlighted' : ''}`}
                    onClick={() => {
                      if (!isActive) {
                        onFork(nodeId);
                      }
                    }}
                    onMouseEnter={(e) => {
                      setHoveredNodeId(nodeId);
                      showTooltip(e, {
                        title: null,
                        body: renderNodeTooltipContent(node, idx),
                        position: 'right',
                      });
                    }}
                    onMouseLeave={() => {
                      setHoveredNodeId(null);
                      hideTooltip();
                    }}
                  >
                    <span className="tree-node-tag">{getNodeIcon(node.kind)} S{idx + 1}</span>
                    <span className="tree-node-label">{truncate(fullLabel)}</span>
                    {!isActive && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '1px 5px', fontSize: '10px', marginLeft: 'auto' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          onFork(nodeId);
                        }}
                        title="Fork new research branch from this intermediate node"
                      >
                        FORK ⑂
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '30px 0' }}>
            NO ACTIVE TREE BRANCH
          </div>
        )}
      </div>

      <TooltipPortal tooltipState={tooltipState} />
    </div>
  );
}
