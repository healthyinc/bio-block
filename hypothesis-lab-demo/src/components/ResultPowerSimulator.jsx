import { useState, useMemo } from 'react';
import { useTooltip, TooltipPortal } from './Tooltip';
import { STAT_TOOLTIPS } from './statDefinitions';

export default function ResultPowerSimulator({ effectSize = 0.5, currentN = 65 }) {
  const [sampleN, setSampleN] = useState(currentN);
  const [effectD, setEffectD] = useState(Math.abs(effectSize) || 0.5);
  const [alpha, setAlpha] = useState(0.05);
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();

  // two-sample t-test power approximation
  function computePower(n, d, sigAlpha) {
    const zAlpha = sigAlpha === 0.01 ? 2.576 : sigAlpha === 0.1 ? 1.645 : 1.96;
    const nEff = n / 2;
    const zBeta = d * Math.sqrt(nEff / 2) - zAlpha;

    // standard normal cdf approx
    const cdf = (z) => 1 / (1 + Math.exp(-0.07056 * Math.pow(z, 3) - 1.5976 * z));
    const power = cdf(zBeta);
    return Math.max(0.05, Math.min(0.999, power));
  }

  // required N for 80% / 90% power
  const requiredN80 = useMemo(() => {
    for (let n = 10; n <= 3000; n += 2) {
      if (computePower(n, effectD, alpha) >= 0.8) return n;
    }
    return 3000;
  }, [effectD, alpha]);

  const requiredN90 = useMemo(() => {
    for (let n = 10; n <= 4000; n += 2) {
      if (computePower(n, effectD, alpha) >= 0.9) return n;
    }
    return 4000;
  }, [effectD, alpha]);

  const currentCalculatedPower = computePower(sampleN, effectD, alpha);
  const powerPct = Math.round(currentCalculatedPower * 100);

  // power curve svg points
  const curvePoints = useMemo(() => {
    const pts = [];
    const maxN = Math.max(250, requiredN90 * 1.3);
    for (let i = 0; i <= 60; i++) {
      const nVal = 10 + (maxN / 60) * i;
      const pow = computePower(nVal, effectD, alpha);
      pts.push({ nVal, pow });
    }
    return { pts, maxN };
  }, [effectD, alpha, requiredN90]);

  const svgWidth = 480;
  const svgHeight = 200;
  const paddingLeft = 45;
  const paddingRight = 20;
  const paddingTop = 25;
  const paddingBottom = 35;

  const plotWidth = svgWidth - paddingLeft - paddingRight;
  const plotHeight = svgHeight - paddingTop - paddingBottom;

  const getX = (nVal) => paddingLeft + (nVal / curvePoints.maxN) * plotWidth;
  const getY = (pow) => paddingTop + plotHeight - pow * plotHeight;

  const pathString = curvePoints.pts
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.nVal).toFixed(1)},${getY(pt.pow).toFixed(1)}`)
    .join(' ');

  const currentPtX = getX(sampleN);
  const currentPtY = getY(currentCalculatedPower);
  const target80Y = getY(0.8);

  return (
    <div style={{ marginTop: '12px', background: 'var(--bg-panel)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-card)', padding: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: 'var(--accent-amber)', letterSpacing: '0.5px' }}>
          INTERACTIVE STATISTICAL POWER & SAMPLE SIZE SIMULATOR
        </div>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: powerPct >= 80 ? 'var(--accent-green)' : 'var(--accent-red)',
            fontWeight: 700,
            cursor: 'help',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
          }}
          onMouseEnter={(e) => showTooltip(e, STAT_TOOLTIPS.power)}
          onMouseLeave={hideTooltip}
        >
          <span>Power (1−β): {powerPct}% {powerPct >= 80 ? '(≥80% — Adequate)' : '(< 80% — Underpowered)'}</span>
          <span className="stat-info-badge" style={{ width: '12px', height: '12px', fontSize: '8.5px' }}>i</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '14px', background: 'var(--bg-card)', padding: '10px', borderRadius: 'var(--radius-sharp)', border: '1px solid var(--border-subtle)' }}>
        <div>
          <label
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '4px',
              cursor: 'help',
            }}
            onMouseEnter={(e) => showTooltip(e, STAT_TOOLTIPS.sample_size)}
            onMouseLeave={hideTooltip}
          >
            <span>SAMPLE SIZE (N = {sampleN}):</span>
            <span className="stat-info-badge" style={{ width: '11px', height: '11px', fontSize: '8px' }}>i</span>
          </label>
          <input
            type="range"
            min="10"
            max={Math.max(300, requiredN90 * 1.2)}
            step="2"
            value={sampleN}
            onChange={(e) => setSampleN(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent-amber)' }}
          />
        </div>

        <div>
          <label
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '4px',
              cursor: 'help',
            }}
            onMouseEnter={(e) => showTooltip(e, STAT_TOOLTIPS.effect_size_d)}
            onMouseLeave={hideTooltip}
          >
            <span>EFFECT SIZE (d = {effectD.toFixed(2)}):</span>
            <span className="stat-info-badge" style={{ width: '11px', height: '11px', fontSize: '8px' }}>i</span>
          </label>
          <input
            type="range"
            min="0.1"
            max="1.5"
            step="0.05"
            value={effectD}
            onChange={(e) => setEffectD(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent-amber)' }}
          />
        </div>

        <div>
          <label
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '4px',
              cursor: 'help',
            }}
            onMouseEnter={(e) => showTooltip(e, STAT_TOOLTIPS.significance_level)}
            onMouseLeave={hideTooltip}
          >
            <span>SIGNIFICANCE LEVEL (α):</span>
            <span className="stat-info-badge" style={{ width: '11px', height: '11px', fontSize: '8px' }}>i</span>
          </label>
          <div style={{ display: 'flex', gap: '4px' }}>
            {[0.01, 0.05, 0.1].map((a) => (
              <button
                key={a}
                style={{
                  flex: 1,
                  background: alpha === a ? 'var(--accent-amber-bg)' : 'transparent',
                  border: `1px solid ${alpha === a ? 'var(--accent-amber)' : 'var(--border-subtle)'}`,
                  color: alpha === a ? 'var(--accent-amber)' : 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10px',
                  borderRadius: '2px',
                  padding: '2px 0',
                  cursor: 'pointer',
                }}
                onClick={() => setAlpha(a)}
              >
                {a}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sharp)', padding: '10px' }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          <line x1={paddingLeft} y1={target80Y} x2={svgWidth - paddingRight} y2={target80Y} stroke="var(--accent-green)" strokeWidth="1" strokeDasharray="3,3" />
          <text x={svgWidth - paddingRight - 4} y={target80Y - 4} fill="var(--accent-green)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="end">
            80% Power Target
          </text>

          <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />
          <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />

          {[0, 0.25, 0.5, 0.75, 1.0].map((v, i) => (
            <g key={i}>
              <text x={paddingLeft - 6} y={getY(v) + 3} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="end">
                {Math.round(v * 100)}%
              </text>
            </g>
          ))}

          <path d={pathString} fill="none" stroke="var(--accent-amber)" strokeWidth="2.5" />

          <circle cx={currentPtX} cy={currentPtY} r="5" fill="var(--accent-green)" stroke="var(--text-primary)" strokeWidth="1.5" />
          <text x={currentPtX} y={currentPtY - 8} fill="var(--accent-green)" fontSize="10" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
            N={sampleN} ({powerPct}%)
          </text>

          <text x={paddingLeft + plotWidth / 2} y={svgHeight - 8} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">
            Sample Size (N)
          </text>
        </svg>

        <div style={{ marginTop: '8px', padding: '6px 10px', background: 'var(--bg-panel)', border: '1px solid var(--border-subtle)', fontFamily: 'var(--font-mono)', fontSize: '11px', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
          <span>Required N for 80% Power: <strong style={{ color: 'var(--accent-green)' }}>N ≈ {requiredN80}</strong></span>
          <span>Required N for 90% Power: <strong style={{ color: 'var(--accent-amber)' }}>N ≈ {requiredN90}</strong></span>
        </div>
      </div>

      <TooltipPortal tooltipState={tooltipState} />
    </div>
  );
}
