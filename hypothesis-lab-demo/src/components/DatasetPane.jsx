import { useState } from 'react';
import BivariateInspector from './BivariateInspector';

function MiniBarChart({ col }) {
  const [tooltip, setTooltip] = useState(null);

  if (col.dtype === 'numeric') {
    const min = col.min_val ?? 0;
    const max = col.max_val ?? 1;
    const mean = col.mean ?? (min + max) / 2;
    const std = col.std ?? (max - min) / 4;

    const q1 = mean - std;
    const q3 = mean + std;
    const bins = [
      { label: `Min: ${min.toFixed(2)} (Lower Bound)`, name: 'MIN', height: 25 },
      { label: `Q1: ~${q1.toFixed(2)} (25th Percentile)`, name: 'Q1', height: 55 },
      { label: `Mean (μ): ${mean.toFixed(2)} ± ${std.toFixed(2)}`, name: 'μ', height: 100 },
      { label: `Q3: ~${q3.toFixed(2)} (75th Percentile)`, name: 'Q3', height: 55 },
      { label: `Max: ${max.toFixed(2)} (Upper Bound)`, name: 'MAX', height: 25 },
    ];

    const maxH = Math.max(...bins.map(b => b.height));

    return (
      <div className="mini-chart-container">
        <div className="mini-chart-bars numeric">
          {bins.map((bin, i) => (
            <div
              key={i}
              className="mini-bar-wrapper"
              onMouseEnter={() => setTooltip(bin.label)}
              onMouseLeave={() => setTooltip(null)}
            >
              <div className="mini-bar-track">
                <div
                  className="mini-bar numeric-bar"
                  style={{ height: `${(bin.height / maxH) * 100}%` }}
                />
              </div>
              <span className="mini-bar-label">{bin.name}</span>
            </div>
          ))}
        </div>
        {tooltip && <div className="mini-chart-tooltip">{tooltip}</div>}
      </div>
    );
  }

  if (col.dtype === 'categorical' && col.top_values && col.top_values.length > 0) {
    const maxCount = Math.max(...col.top_values.map(v => v.count || 0));
    const totalCount = col.top_values.reduce((sum, v) => sum + (v.count || 0), 0);

    return (
      <div className="mini-chart-container">
        <div className="mini-chart-bars categorical">
          {col.top_values.slice(0, 6).map((val, i) => {
            const count = val.count || 0;
            const pct = totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : '0';
            const valLabel = String(val.value ?? 'N/A');
            const hoverText = `${valLabel}: ${count.toLocaleString()} (${pct}%)`;

            return (
              <div
                key={i}
                className="mini-bar-wrapper"
                onMouseEnter={() => setTooltip(hoverText)}
                onMouseLeave={() => setTooltip(null)}
              >
                <div className="mini-bar-track">
                  <div
                    className="mini-bar categorical-bar"
                    style={{ height: `${Math.max((count / maxCount) * 100, 10)}%` }}
                  />
                </div>
                <span className="mini-bar-label" title={valLabel}>
                  {valLabel.length > 5 ? valLabel.substring(0, 5) + '…' : valLabel}
                </span>
              </div>
            );
          })}
        </div>
        {tooltip && <div className="mini-chart-tooltip">{tooltip}</div>}
      </div>
    );
  }

  if (col.dtype === 'boolean') {
    const truePct = col.missing_pct != null ? (100 - col.missing_pct) : 50;
    const falsePct = 100 - truePct;

    return (
      <div className="mini-chart-container">
        <div className="mini-chart-bars boolean">
          <div
            className="mini-bar-wrapper"
            onMouseEnter={() => setTooltip(`True: ~${truePct.toFixed(1)}%`)}
            onMouseLeave={() => setTooltip(null)}
          >
            <div className="mini-bar-track">
              <div className="mini-bar boolean-true-bar" style={{ height: `${truePct}%` }} />
            </div>
            <span className="mini-bar-label">TRUE</span>
          </div>
          <div
            className="mini-bar-wrapper"
            onMouseEnter={() => setTooltip(`False: ~${falsePct.toFixed(1)}%`)}
            onMouseLeave={() => setTooltip(null)}
          >
            <div className="mini-bar-track">
              <div className="mini-bar boolean-false-bar" style={{ height: `${falsePct}%` }} />
            </div>
            <span className="mini-bar-label">FALSE</span>
          </div>
        </div>
        {tooltip && <div className="mini-chart-tooltip">{tooltip}</div>}
      </div>
    );
  }

  return null;
}

function isIdentifier(c) {
  if (!c) return false;
  if (c.dtype === 'identifier') return true;
  const name = c.name.toLowerCase().trim().replace(/_/g, ' ').replace(/-/g, ' ');
  const patterns = ['patient #', 'patient id', 'pt #', 'pt id', 'subject id', 'participant id', 'sample id', 'first name', 'last name', 'full name', 'mrn', 'ssn', 'uuid'];
  if (patterns.some(p => name.includes(p))) return true;
  if (['id', 'row', 'index', 'seq', '#', 'number', 'patient'].includes(name)) return true;
  if (name.endsWith(' id') || name.endsWith(' #')) return true;
  return false;
}

export default function DatasetPane({ profile }) {
  const [activeTab, setActiveTab] = useState('schema');
  const [showIdentifiers, setShowIdentifiers] = useState(false);

  if (!profile) return null;

  const analyticalColumns = profile.columns.filter(c => !isIdentifier(c));
  const identifierColumns = profile.columns.filter(c => isIdentifier(c));
  const displayColumns = analyticalColumns.length > 0 ? analyticalColumns : profile.columns;

  return (
    <div className="pane dataset-pane">
      <div className="pane-header">
        <h2>
          <span className="pane-header-num">01</span>
          VARIABLE SCHEMA
        </h2>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            className={`btn btn-sm ${activeTab === 'schema' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setActiveTab('schema')}
          >
            SCHEMA
          </button>
          <button
            className={`btn btn-sm ${activeTab === 'bivariate' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setActiveTab('bivariate')}
          >
            BIVARIATE
          </button>
        </div>
      </div>

      <div className="pane-body">
        <div className="dataset-summary">
          <div className="dataset-stat">
            <span className="dataset-stat-label">OBSERVATIONS</span>
            <span className="dataset-stat-value">{profile.row_count.toLocaleString()}</span>
          </div>
          <div className="dataset-stat">
            <span className="dataset-stat-label">VARIABLES</span>
            <span className="dataset-stat-value">{profile.column_count}</span>
          </div>
        </div>

        {activeTab === 'bivariate' ? (
          <BivariateInspector profile={profile} />
        ) : (
          <>
            <div className="section-label">Profiled Features ({displayColumns.length})</div>
            {displayColumns.map((col) => (
              <div key={col.name} className="glass-card column-card">
                <div className="column-card-header">
                  <span className="column-name">{col.name}</span>
                  <span className={`column-type-badge ${col.dtype}`}>
                    {col.dtype}
                  </span>
                </div>

                <MiniBarChart col={col} />

                <div className="column-stats">
                  {col.dtype === 'numeric' && (
                    <>
                      {col.mean != null && <span>μ=<strong>{col.mean.toFixed(1)}</strong></span>}
                      {col.std != null && <span>σ=<strong>{col.std.toFixed(1)}</strong></span>}
                      {col.min_val != null && <span>min=<strong>{col.min_val.toFixed(1)}</strong></span>}
                      {col.max_val != null && <span>max=<strong>{col.max_val.toFixed(1)}</strong></span>}
                    </>
                  )}
                  {col.dtype === 'categorical' && col.cardinality != null && (
                    <span>n_unique=<strong>{col.cardinality}</strong></span>
                  )}
                </div>

                {col.missing_pct > 0 && (
                  <div style={{ marginTop: '4px', fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--accent-red)' }}>
                    {col.missing_pct}% missing values
                  </div>
                )}
              </div>
            ))}

            {identifierColumns.length > 0 && analyticalColumns.length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ width: '100%', fontSize: '10px', fontFamily: 'var(--font-mono)' }}
                  onClick={() => setShowIdentifiers(!showIdentifiers)}
                >
                  {showIdentifiers ? '▲ HIDE EXCLUDED IDENTIFIERS' : `▼ SHOW EXCLUDED IDENTIFIERS (${identifierColumns.length})`}
                </button>

                {showIdentifiers && (
                  <div style={{ marginTop: '8px', opacity: 0.7 }}>
                    {identifierColumns.map((col) => (
                      <div key={col.name} className="glass-card column-card" style={{ borderStyle: 'dashed' }}>
                        <div className="column-card-header">
                          <span className="column-name">{col.name}</span>
                          <span className="column-type-badge text" style={{ background: 'rgba(239,68,68,0.15)', color: '#EF4444' }}>
                            IDENTIFIER (EXCLUDED)
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
