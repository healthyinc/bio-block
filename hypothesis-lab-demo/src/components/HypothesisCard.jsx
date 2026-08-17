export default function HypothesisCard({ hypothesis, onRunAnalysis, analyses, analysisResult }) {
  if (!hypothesis) return null;

  const hasResult = analysisResult && analysisResult.hypothesis_id === hypothesis.id;
  const r = analysisResult?.result || {};
  const isSignificant = hasResult && (r.significant === true || r.p_value < 0.05);

  // suggested tests first
  const sortedAnalyses = [...(analyses || [])].sort((a, b) => {
    if (a.is_suggested && !b.is_suggested) return -1;
    if (!a.is_suggested && b.is_suggested) return 1;
    return (a.display_name || '').localeCompare(b.display_name || '');
  });

  return (
    <div className="glass-card hypothesis-card">
      <div className="hypothesis-statement">{hypothesis.statement}</div>

      <div className="hypothesis-details-container">
        {hypothesis.null_hypothesis && (
          <div className="hypothesis-row">
            <div className="hypothesis-formula">H₀ (NULL HYPOTHESIS)</div>
            <div className="hypothesis-text">{hypothesis.null_hypothesis}</div>
          </div>
        )}

        {hypothesis.alternative_hypothesis && (
          <div className="hypothesis-row">
            <div className="hypothesis-formula">H₁ (ALTERNATIVE HYPOTHESIS)</div>
            <div className="hypothesis-text">{hypothesis.alternative_hypothesis}</div>
          </div>
        )}
      </div>

      {sortedAnalyses.length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <div className="section-label">APPLICABLE TESTS ({sortedAnalyses.length})</div>
          {sortedAnalyses.map((analysis) => (
            <div
              key={analysis.id}
              className={`analysis-card ${analysis.is_suggested ? 'suggested' : ''}`}
              onClick={() => onRunAnalysis(hypothesis.id, analysis.id)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <div className="analysis-name">{analysis.display_name}</div>
                {analysis.is_suggested && (
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '10px',
                    fontWeight: 700,
                    color: 'var(--accent-amber)',
                    background: 'var(--accent-amber-bg)',
                    border: '1px solid var(--accent-amber-border)',
                    padding: '2px 6px',
                    borderRadius: '2px',
                    letterSpacing: '0.5px',
                  }}>
                    RECOMMENDED
                  </span>
                )}
              </div>

              <div className="analysis-description">{analysis.description}</div>

              {analysis.suggestion_reason && (
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  color: 'var(--accent-amber)',
                  background: 'rgba(217, 119, 6, 0.06)',
                  borderLeft: '2px solid var(--accent-amber)',
                  padding: '4px 8px',
                  margin: '6px 0 8px 0',
                }}>
                  ↳ Reason: {analysis.suggestion_reason}
                </div>
              )}

              {analysis.tradeoffs && analysis.tradeoffs.length > 0 && (
                <div style={{ margin: '6px 0', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {analysis.tradeoffs.map((t, i) => (
                    <div key={i} style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                      <strong style={{ color: 'var(--text-secondary)' }}>{t.label}:</strong> {t.description}
                    </div>
                  ))}
                </div>
              )}

              <button className="btn btn-primary btn-sm" style={{ marginTop: '6px' }}>
                EXECUTE {analysis.display_name.toUpperCase()}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
