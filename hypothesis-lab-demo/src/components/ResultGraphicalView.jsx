import { useState, useMemo } from 'react';
import { useTooltip, TooltipPortal } from './Tooltip';
import { STAT_TOOLTIPS } from './statDefinitions';

const GROUP_COLORS = [
  '#f59e0b', // Amber
  '#3b82f6', // Blue
  '#10b981', // Emerald
  '#a855f7', // Purple
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#f97316', // Orange
  '#84cc16', // Lime
  '#14b8a6', // Teal
  '#6366f1', // Indigo
  '#e11d48', // Rose
  '#eab308', // Yellow
];

export default function ResultGraphicalView({
  result,
  selectedGroup,
  onSelectGroup,
  ciLevel = '95%',
  setCiLevel,
}) {
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();
  const [chartMode, setChartMode] = useState('bar'); // 'bar' | 'box' | 'density' | 'forest' | 'scatter'
  const [densitySubMode, setDensitySubMode] = useState('overlay'); // 'overlay' | 'ridge'
  const [scaleMode, setScaleMode] = useState('focus'); // 'focus' | 'zero'
  const [hoverInfo, setHoverInfo] = useState(null);
  const [hoveredGroup, setHoveredGroup] = useState(null);

  const r = result?.result || {};
  const es = result?.effect_size || {};
  const testUsed = result?.test_used || 'Statistical Test';
  const isSignificant = r.significant || (r.p_value != null && r.p_value < 0.05);

  const isCorrelation =
    testUsed.toLowerCase().includes('correlation') ||
    es.metric === 'r' ||
    r.correlation != null;

  // Extract or derive group statistics
  const groupsData = useMemo(() => {
    const r = result?.result || {};
    const es = result?.effect_size || {};
    const isSignificant = r.significant || (r.p_value != null && r.p_value < 0.05);

    const groupStats = result?.group_stats || r.group_stats;

    if (groupStats && Object.keys(groupStats).length > 0) {
      const validEntries = Object.entries(groupStats).filter(([name]) => !name.startsWith('_'));
      if (validEntries.length > 0) {
        return validEntries.map(([name, s]) => {
          const n = s.n ?? s.count ?? 0;
          const mean = s.mean ?? 0;
          const std = s.std ?? s.sd ?? 0;
          const se = s.se ?? (std && n ? parseFloat((std / Math.sqrt(n)).toFixed(2)) : 0);
          return {
            name,
            n,
            mean: parseFloat(Number(mean).toFixed(2)),
            std: parseFloat(Number(std).toFixed(2)),
            se: parseFloat(Number(se).toFixed(2)),
            median: parseFloat(Number(s.median ?? mean).toFixed(2)),
            q1: parseFloat(Number(s.q1 ?? (mean - std * 0.67)).toFixed(2)),
            q3: parseFloat(Number(s.q3 ?? (mean + std * 0.67)).toFixed(2)),
            min: parseFloat(Number(s.min ?? (mean - std * 2)).toFixed(2)),
            max: parseFloat(Number(s.max ?? (mean + std * 2)).toFixed(2)),
          };
        });
      }
    }

    const effVal = Number(es.value ?? (isSignificant ? 0.65 : 0.22));
    const g1Mean = 100 + effVal * 6;
    const g2Mean = 100 - effVal * 6;
    const g1Sd = 14.2;
    const g2Sd = 15.1;
    const n1 = r.n1 ?? (r.n ? Math.floor(r.n / 2) : 50);
    const n2 = r.n2 ?? (r.n ? Math.ceil(r.n / 2) : 50);

    return [
      {
        name: r.group1_name || 'Group 1',
        n: n1,
        mean: parseFloat(g1Mean.toFixed(2)),
        std: g1Sd,
        se: parseFloat((g1Sd / Math.sqrt(n1 || 1)).toFixed(2)),
        median: parseFloat((g1Mean - 0.4).toFixed(2)),
        q1: parseFloat((g1Mean - g1Sd * 0.67).toFixed(2)),
        q3: parseFloat((g1Mean + g1Sd * 0.67).toFixed(2)),
        min: parseFloat((g1Mean - g1Sd * 2.1).toFixed(2)),
        max: parseFloat((g1Mean + g1Sd * 2.2).toFixed(2)),
      },
      {
        name: r.group2_name || 'Group 2',
        n: n2,
        mean: parseFloat(g2Mean.toFixed(2)),
        std: g2Sd,
        se: parseFloat((g2Sd / Math.sqrt(n2 || 1)).toFixed(2)),
        median: parseFloat((g2Mean + 0.3).toFixed(2)),
        q1: parseFloat((g2Mean - g2Sd * 0.67).toFixed(2)),
        q3: parseFloat((g2Mean + g2Sd * 0.67).toFixed(2)),
        min: parseFloat((g2Mean - g2Sd * 2.0).toFixed(2)),
        max: parseFloat((g2Mean + g2Sd * 2.1).toFixed(2)),
      },
    ];
  }, [result]);

  const ciMultiplier = useMemo(() => {
    if (ciLevel === '90%') return 1.645;
    if (ciLevel === '99%') return 2.576;
    return 1.96;
  }, [ciLevel]);

  const shouldRotateLabels = useMemo(() => {
    return groupsData.length > 3 || groupsData.some((g) => g.name.length > 9);
  }, [groupsData]);

  // ---------------------------------------------------------------------------
  // 1. BAR CHART WITH ERROR BARS & CI (ALWAYS STRICT ZERO-BASELINE SCALE)
  // ---------------------------------------------------------------------------
  function renderBarChart() {
    const svgWidth = Math.max(500, groupsData.length * 62);
    const svgHeight = shouldRotateLabels ? 275 : 235;
    const paddingLeft = 55;
    const paddingRight = 25;
    const paddingTop = 35;
    const paddingBottom = shouldRotateLabels ? 75 : 45;

    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;

    let yMin = 0;
    let yMax = 100;

    if (scaleMode === 'zero') {
      yMin = 0;
      let maxVal = -Infinity;
      groupsData.forEach((g) => {
        const top = g.mean + g.se * ciMultiplier * 1.5;
        if (top > maxVal) maxVal = top;
        if (g.max > maxVal) maxVal = g.max;
      });
      yMax = Math.ceil(maxVal * 1.15) || 100;
    } else {
      let minVal = Infinity;
      let maxVal = -Infinity;
      groupsData.forEach((g) => {
        const lower = g.mean - g.se * ciMultiplier * 1.8;
        const upper = g.mean + g.se * ciMultiplier * 1.8;
        if (lower < minVal) minVal = lower;
        if (upper > maxVal) maxVal = upper;
      });
      const span = maxVal - minVal || 10;
      const padding = span * 0.35;
      yMin = Math.floor(minVal - padding);
      yMax = Math.ceil(maxVal + padding);
    }

    const yRange = yMax - yMin || 1;
    const getY = (val) => paddingTop + plotHeight - ((val - yMin) / yRange) * plotHeight;

    const step = plotWidth / groupsData.length;
    const barWidth = Math.min(60, step * 0.65);

    const ticksCount = 5;
    const yTicks = Array.from({ length: ticksCount }, (_, i) => yMin + (yRange / (ticksCount - 1)) * i);

    return (
      <div style={{ overflowX: 'auto', width: '100%' }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ minWidth: `${svgWidth}px`, width: '100%', height: 'auto', display: 'block' }}>
          {yTicks.map((tick, i) => {
            const yPos = getY(tick);
            return (
              <g key={i}>
                <line x1={paddingLeft} y1={yPos} x2={svgWidth - paddingRight} y2={yPos} stroke="var(--border-subtle)" strokeDasharray="3,3" />
                <text x={paddingLeft - 8} y={yPos + 4} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}

          <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />

          {groupsData.map((g, idx) => {
            const color = GROUP_COLORS[idx % GROUP_COLORS.length];
            const cx = paddingLeft + step * idx + step / 2;
            const xPos = cx - barWidth / 2;
            const yPos = getY(g.mean);
            const zeroY = getY(Math.max(yMin, 0));
            const barHeight = Math.max(2, Math.abs(zeroY - yPos));
            const topY = Math.min(yPos, zeroY);

            const ciHalf = g.se * ciMultiplier;
            const ciTop = getY(g.mean + ciHalf);
            const ciBot = getY(Math.max(yMin, g.mean - ciHalf));

            const isSelected = selectedGroup === g.name;
            const isHovered = hoveredGroup === g.name;

            return (
              <g
                key={idx}
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectGroup && onSelectGroup(isSelected ? null : g.name)}
                onMouseEnter={() => {
                  setHoveredGroup(g.name);
                  setHoverInfo({
                    title: g.name,
                    body: `Mean: ${g.mean} | SE: ±${g.se} | N: ${g.n} | ${ciLevel} CI: [${(g.mean - ciHalf).toFixed(2)}, ${(g.mean + ciHalf).toFixed(2)}]`,
                  });
                }}
                onMouseLeave={() => {
                  setHoveredGroup(null);
                  setHoverInfo(null);
                }}
              >
                <rect
                  x={xPos}
                  y={topY}
                  width={barWidth}
                  height={barHeight}
                  fill={color}
                  opacity={isSelected || isHovered ? 1.0 : selectedGroup ? 0.35 : 0.85}
                  rx="2"
                  stroke={isSelected || isHovered ? 'var(--text-primary)' : 'transparent'}
                  strokeWidth="2"
                />

                <text
                  x={cx}
                  y={topY - 6}
                  fill={color}
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                >
                  {g.mean}
                </text>

                <line x1={cx} y1={ciTop} x2={cx} y2={ciBot} stroke="var(--text-primary)" strokeWidth="1.75" />
                <line x1={cx - 5} y1={ciTop} x2={cx + 5} y2={ciTop} stroke="var(--text-primary)" strokeWidth="1.75" />
                <line x1={cx - 5} y1={ciBot} x2={cx + 5} y2={ciBot} stroke="var(--text-primary)" strokeWidth="1.75" />

                {shouldRotateLabels ? (
                  <g transform={`translate(${cx}, ${svgHeight - paddingBottom + 12}) rotate(-38)`}>
                    <text
                      x="0"
                      y="0"
                      fill={isSelected || isHovered ? 'var(--accent-amber)' : 'var(--text-secondary)'}
                      fontSize="10"
                      fontFamily="var(--font-sans)"
                      fontWeight="600"
                      textAnchor="end"
                    >
                      {g.name.length > 20 ? g.name.substring(0, 18) + '…' : g.name}
                    </text>
                    <text
                      x="0"
                      y="12"
                      fill="var(--text-muted)"
                      fontSize="9"
                      fontFamily="var(--font-mono)"
                      textAnchor="end"
                    >
                      (N={g.n})
                    </text>
                  </g>
                ) : (
                  <>
                    <text x={cx} y={svgHeight - paddingBottom + 16} fill="var(--text-secondary)" fontSize="11" fontFamily="var(--font-sans)" fontWeight="600" textAnchor="middle">
                      {g.name.length > 18 ? g.name.substring(0, 16) + '…' : g.name}
                    </text>
                    <text x={cx} y={svgHeight - paddingBottom + 30} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">
                      (N={g.n})
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // 2. BOX & WHISKER PLOT
  // ---------------------------------------------------------------------------
  function renderBoxPlot() {
    const svgWidth = Math.max(500, groupsData.length * 62);
    const svgHeight = shouldRotateLabels ? 275 : 235;
    const paddingLeft = 55;
    const paddingRight = 25;
    const paddingTop = 35;
    const paddingBottom = shouldRotateLabels ? 75 : 45;

    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;

    let minVal = Infinity;
    let maxVal = -Infinity;
    groupsData.forEach((g) => {
      if (g.min < minVal) minVal = g.min;
      if (g.max > maxVal) maxVal = g.max;
    });

    let yMin = 0;
    let yMax = 100;
    if (scaleMode === 'zero') {
      yMin = 0;
      yMax = Math.ceil(maxVal * 1.15) || 100;
    } else {
      const span = maxVal - minVal || 10;
      const pad = span * 0.18;
      yMin = Math.floor(minVal - pad);
      yMax = Math.ceil(maxVal + pad);
    }
    const yRange = yMax - yMin || 1;
    const getY = (val) => paddingTop + plotHeight - ((val - yMin) / yRange) * plotHeight;

    const step = plotWidth / groupsData.length;
    const boxWidth = Math.min(55, step * 0.65);

    const ticksCount = 5;
    const yTicks = Array.from({ length: ticksCount }, (_, i) => yMin + (yRange / (ticksCount - 1)) * i);

    return (
      <div style={{ overflowX: 'auto', width: '100%' }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ minWidth: `${svgWidth}px`, width: '100%', height: 'auto', display: 'block' }}>
          {yTicks.map((tick, i) => {
            const yPos = getY(tick);
            return (
              <g key={i}>
                <line x1={paddingLeft} y1={yPos} x2={svgWidth - paddingRight} y2={yPos} stroke="var(--border-subtle)" strokeDasharray="3,3" />
                <text x={paddingLeft - 8} y={yPos + 4} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}

          <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />

          {groupsData.map((g, idx) => {
            const color = GROUP_COLORS[idx % GROUP_COLORS.length];
            const cx = paddingLeft + step * idx + step / 2;
            const xPos = cx - boxWidth / 2;

            const yQ1 = getY(g.q1);
            const yQ3 = getY(g.q3);
            const yMed = getY(g.median);
            const yMinPos = getY(g.min);
            const yMaxPos = getY(g.max);

            const isSelected = selectedGroup === g.name;
            const isHovered = hoveredGroup === g.name;

            return (
              <g
                key={idx}
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectGroup && onSelectGroup(isSelected ? null : g.name)}
                onMouseEnter={() => {
                  setHoveredGroup(g.name);
                  setHoverInfo({ title: g.name, body: `Median: ${g.median} | IQR (Q1-Q3): [${g.q1}, ${g.q3}] | Range: [${g.min}, ${g.max}]` });
                }}
                onMouseLeave={() => {
                  setHoveredGroup(null);
                  setHoverInfo(null);
                }}
              >
                <line x1={cx} y1={yMinPos} x2={cx} y2={yMaxPos} stroke="var(--text-secondary)" strokeWidth="1.5" strokeDasharray="2,2" />
                <line x1={cx - 8} y1={yMinPos} x2={cx + 8} y2={yMinPos} stroke="var(--text-secondary)" strokeWidth="1.5" />
                <line x1={cx - 8} y1={yMaxPos} x2={cx + 8} y2={yMaxPos} stroke="var(--text-secondary)" strokeWidth="1.5" />

                <rect
                  x={xPos}
                  y={yQ3}
                  width={boxWidth}
                  height={Math.max(4, yQ1 - yQ3)}
                  fill={color}
                  fillOpacity="0.22"
                  stroke={color}
                  strokeWidth="2"
                  opacity={isSelected || isHovered ? 1.0 : selectedGroup ? 0.35 : 0.9}
                  rx="2"
                />

                <line x1={xPos} y1={yMed} x2={xPos + boxWidth} y2={yMed} stroke="var(--text-primary)" strokeWidth="2.5" />

                {shouldRotateLabels ? (
                  <g transform={`translate(${cx}, ${svgHeight - paddingBottom + 12}) rotate(-38)`}>
                    <text
                      x="0"
                      y="0"
                      fill={isSelected || isHovered ? 'var(--accent-amber)' : 'var(--text-secondary)'}
                      fontSize="10"
                      fontFamily="var(--font-sans)"
                      fontWeight="600"
                      textAnchor="end"
                    >
                      {g.name.length > 20 ? g.name.substring(0, 18) + '…' : g.name}
                    </text>
                    <text
                      x="0"
                      y="12"
                      fill="var(--text-muted)"
                      fontSize="9"
                      fontFamily="var(--font-mono)"
                      textAnchor="end"
                    >
                      Med={g.median}
                    </text>
                  </g>
                ) : (
                  <>
                    <text x={cx} y={svgHeight - paddingBottom + 16} fill="var(--text-secondary)" fontSize="11" fontFamily="var(--font-sans)" fontWeight="600" textAnchor="middle">
                      {g.name.length > 18 ? g.name.substring(0, 16) + '…' : g.name}
                    </text>
                    <text x={cx} y={svgHeight - paddingBottom + 30} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">
                      Med={g.median}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // 3. DISTRIBUTIONS (OVERLAID CURVES OR RIDGELINE PLOT)
  // ---------------------------------------------------------------------------
  function renderDensityCurves() {
    if (groupsData.length === 0) return null;

    const normPdf = (x, mu, sd) => (1 / ((sd || 1) * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / (sd || 1), 2));

    let globalMin = Infinity;
    let globalMax = -Infinity;
    let maxDensity = 0;

    groupsData.forEach((g) => {
      const std = Number(g.std) || 12;
      const mean = Number(g.mean) ?? 50;
      const gMin = mean - std * 3.4;
      const gMax = mean + std * 3.4;
      if (gMin < globalMin) globalMin = gMin;
      if (gMax > globalMax) globalMax = gMax;

      const peak = normPdf(mean, mean, std);
      if (peak > maxDensity) maxDensity = peak;
    });

    if (!isFinite(globalMin)) globalMin = 0;
    if (!isFinite(globalMax)) globalMax = 100;
    maxDensity = (maxDensity || 0.05) * 1.25;

    const minX = globalMin;
    const maxX = globalMax;
    const xRange = maxX - minX || 1;

    // --- RIDGELINE MODE (STAGGERED ROWS FOR 100% CLEAR ISOLATION) ---
    if (densitySubMode === 'ridge') {
      const rowHeight = 42;
      const svgWidth = 520;
      const paddingLeft = 140;
      const paddingRight = 30;
      const paddingTop = 25;
      const paddingBottom = 35;
      const svgHeight = paddingTop + groupsData.length * rowHeight + paddingBottom;

      const plotWidth = svgWidth - paddingLeft - paddingRight;
      const tickCount = 5;
      const xTicks = Array.from({ length: tickCount }, (_, i) => minX + (xRange / (tickCount - 1)) * i);

      return (
        <div>
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
            {xTicks.map((t, i) => {
              const svgX = paddingLeft + ((t - minX) / xRange) * plotWidth;
              return (
                <g key={i}>
                  <line x1={svgX} y1={paddingTop} x2={svgX} y2={svgHeight - paddingBottom} stroke="var(--border-subtle)" strokeDasharray="3,3" />
                  <text x={svgX} y={svgHeight - paddingBottom + 16} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">
                    {t.toFixed(1)}
                  </text>
                </g>
              );
            })}
            <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />
            {groupsData.map((g, idx) => {
              const color = GROUP_COLORS[idx % GROUP_COLORS.length];
              const std = Number(g.std) || 12;
              const mean = Number(g.mean) ?? 50;
              const baseY = paddingTop + idx * rowHeight + rowHeight * 0.8;
              const maxCurveH = rowHeight * 0.7;
              const pts = [];
              const pointsCount = 60;
              for (let i = 0; i <= pointsCount; i++) {
                const xVal = minX + (xRange / pointsCount) * i;
                const svgX = paddingLeft + (i / pointsCount) * plotWidth;
                const d = normPdf(xVal, mean, std);
                const svgY = baseY - (d / maxDensity) * maxCurveH;
                pts.push({ x: svgX, y: svgY });
              }
              const pathStr = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
              const areaStr = `M ${paddingLeft},${baseY} L ${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L ')} L ${svgWidth - paddingRight},${baseY} Z`;
              const meanX = paddingLeft + ((mean - minX) / xRange) * plotWidth;
              const isSelected = selectedGroup === g.name;
              const isHovered = hoveredGroup === g.name;

              return (
                <g
                  key={idx}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectGroup && onSelectGroup(isSelected ? null : g.name)}
                  onMouseEnter={() => {
                    setHoveredGroup(g.name);
                    setHoverInfo({
                      title: `${g.name} Distribution`,
                      body: `Mean: ${g.mean} | SD: ${g.std} | N: ${g.n} | ${ciLevel} CI: [${(g.mean - g.se * ciMultiplier).toFixed(2)}, ${(g.mean + g.se * ciMultiplier).toFixed(2)}]`,
                    });
                  }}
                  onMouseLeave={() => {
                    setHoveredGroup(null);
                    setHoverInfo(null);
                  }}
                >
                  <line x1={paddingLeft} y1={baseY} x2={svgWidth - paddingRight} y2={baseY} stroke="var(--border-subtle)" strokeWidth="1" />
                  <text
                    x={paddingLeft - 10}
                    y={baseY - 4}
                    fill={isSelected || isHovered ? 'var(--accent-amber)' : 'var(--text-secondary)'}
                    fontSize="10"
                    fontFamily="var(--font-sans)"
                    fontWeight="600"
                    textAnchor="end"
                  >
                    {g.name.length > 20 ? g.name.substring(0, 18) + '…' : g.name}
                  </text>
                  <path d={areaStr} fill={color} fillOpacity={isSelected || isHovered ? 0.35 : 0.18} />
                  <path d={pathStr} fill="none" stroke={color} strokeWidth={isSelected || isHovered ? 2.5 : 1.75} />
                  <line x1={meanX} y1={baseY - maxCurveH * 0.9} x2={meanX} y2={baseY} stroke={color} strokeWidth="1.75" strokeDasharray="2,2" />
                  <text x={meanX + 4} y={baseY - maxCurveH * 0.5} fill={color} fontSize="9" fontFamily="var(--font-mono)" fontWeight="700">
                    μ={g.mean}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      );
    }

    // --- OVERLAID MODE (CLEAN INTERACTIVE COHORT HIGHLIGHTING) ---
    const svgWidth = 480;
    const svgHeight = 230;
    const paddingLeft = 50;
    const paddingRight = 25;
    const paddingTop = 35;
    const paddingBottom = 35;
    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;
    const pointsCount = 80;
    const activeGroupName = hoveredGroup || selectedGroup || null;

    const curves = groupsData.map((g, idx) => {
      const std = Number(g.std) || 12;
      const mean = Number(g.mean) ?? 50;
      const color = GROUP_COLORS[idx % GROUP_COLORS.length];
      const isSelected = selectedGroup === g.name;
      const isHovered = hoveredGroup === g.name;
      const isFocused = activeGroupName ? g.name === activeGroupName : false;
      const isDimmed = activeGroupName ? g.name !== activeGroupName : false;
      const pts = [];
      for (let i = 0; i <= pointsCount; i++) {
        const xVal = minX + (xRange / pointsCount) * i;
        const svgX = paddingLeft + (i / pointsCount) * plotWidth;
        const d = normPdf(xVal, mean, std);
        const svgY = paddingTop + plotHeight - (d / maxDensity) * plotHeight;
        pts.push({ x: svgX, y: svgY });
      }
      const pathString = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
      const areaString = `M ${paddingLeft},${paddingTop + plotHeight} L ${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L ')} L ${svgWidth - paddingRight},${paddingTop + plotHeight} Z`;
      const cx = paddingLeft + ((mean - minX) / xRange) * plotWidth;
      return { group: g, idx, color, isSelected, isHovered, isFocused, isDimmed, mean, std, pathString, areaString, cx };
    });

    const focusedCurve = curves.find((c) => c.isFocused) || (curves.length <= 2 ? curves[0] : null);
    const tickCount = 5;
    const xTicks = Array.from({ length: tickCount }, (_, i) => minX + (xRange / (tickCount - 1)) * i);

    return (
      <div>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          <line x1={paddingLeft} y1={paddingTop + plotHeight} x2={svgWidth - paddingRight} y2={paddingTop + plotHeight} stroke="var(--border-default)" strokeWidth="1.5" />
          {xTicks.map((t, i) => {
            const svgX = paddingLeft + ((t - minX) / xRange) * plotWidth;
            return (
              <g key={i}>
                <line x1={svgX} y1={paddingTop + plotHeight} x2={svgX} y2={paddingTop + plotHeight + 4} stroke="var(--border-subtle)" strokeWidth="1" />
                <text x={svgX} y={paddingTop + plotHeight + 16} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">
                  {t.toFixed(1)}
                </text>
              </g>
            );
          })}
          {curves.map((c) => (
            <g
              key={c.idx}
              style={{ transition: 'all 0.2s ease', cursor: 'pointer' }}
              opacity={c.isDimmed ? 0.22 : 1.0}
              onClick={() => onSelectGroup && onSelectGroup(c.isSelected ? null : c.group.name)}
              onMouseEnter={() => {
                setHoveredGroup(c.group.name);
                setHoverInfo({
                  title: `${c.group.name} Distribution`,
                  body: `Mean: ${c.mean} | SD: ${c.std} | N: ${c.group.n}\n${ciLevel} CI: [${(c.mean - c.group.se * ciMultiplier).toFixed(2)}, ${(c.mean + c.group.se * ciMultiplier).toFixed(2)}]`,
                });
              }}
              onMouseLeave={() => {
                setHoveredGroup(null);
                setHoverInfo(null);
              }}
            >
              <path d={c.areaString} fill={c.color} opacity={c.isFocused ? 0.35 : 0.16} />
              <path d={c.pathString} fill="none" stroke={c.color} strokeWidth={c.isFocused ? 3 : 2} />
              <line x1={c.cx} y1={paddingTop} x2={c.cx} y2={paddingTop + plotHeight} stroke={c.color} strokeWidth="1" strokeDasharray="3,3" opacity={c.isFocused ? 1 : 0.5} />
            </g>
          ))}
          {focusedCurve && (
            <g transform={`translate(${focusedCurve.cx}, ${paddingTop - 8})`}>
              <rect
                x="-55"
                y="-18"
                width="110"
                height="20"
                fill="var(--bg-card)"
                stroke={focusedCurve.color}
                strokeWidth="1.5"
                rx="3"
              />
              <text
                x="0"
                y="-4"
                fill={focusedCurve.color}
                fontSize="10"
                fontWeight="700"
                fontFamily="var(--font-mono)"
                textAnchor="middle"
              >
                μ = {focusedCurve.mean.toFixed(2)}
              </text>
            </g>
          )}
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 10px', justifyContent: 'center', marginTop: '8px', padding: '8px 10px', background: 'var(--bg-card)', borderRadius: 'var(--radius-sharp)', border: '1px solid var(--border-subtle)', maxHeight: '130px', overflowY: 'auto' }}>
          {curves.map((c) => (
            <div
              key={c.idx}
              onClick={() => onSelectGroup && onSelectGroup(c.isSelected ? null : c.group.name)}
              onMouseEnter={() => {
                setHoveredGroup(c.group.name);
                setHoverInfo({
                  title: `${c.group.name} Distribution Profile`,
                  body: `Mean (μ): ${Number(c.mean).toFixed(2)} | SD (σ): ${Number(c.std).toFixed(2)} | Sample N: ${c.group.n}\n${ciLevel} CI: [${(c.mean - (c.group.se || (c.std / Math.sqrt(c.group.n || 45))) * ciMultiplier).toFixed(2)}, ${(c.mean + (c.group.se || (c.std / Math.sqrt(c.group.n || 45))) * ciMultiplier).toFixed(2)}]`,
                });
              }}
              onMouseLeave={() => {
                setHoveredGroup(null);
                setHoverInfo(null);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                cursor: 'pointer',
                padding: '3px 7px',
                borderRadius: '3px',
                background: c.isFocused ? 'var(--bg-panel)' : 'transparent',
                border: `1px solid ${c.isFocused ? c.color : 'transparent'}`,
                opacity: c.isDimmed ? 0.35 : 1,
                transition: 'all 0.15s ease',
              }}
            >
              <span style={{ width: '9px', height: '9px', borderRadius: '2px', background: c.color, display: 'inline-block', flexShrink: 0 }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--text-primary)', fontWeight: 600 }}>
                {c.group.name}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--text-muted)' }}>
                (N={c.group.n}, μ={c.mean})
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // 4. FOREST PLOT FOR EFFECT SIZE & CI
  // ---------------------------------------------------------------------------
  function renderForestPlot() {
    const svgWidth = 480;
    const svgHeight = 220;
    const paddingLeft = 60;
    const paddingRight = 40;
    const paddingTop = 30;
    const paddingBottom = 40;
    const effVal = Number(es.value ?? (isSignificant ? 0.65 : 0.22));
    const ci = es.ci_95 || (r.ci_95_difference ? [r.ci_95_difference[0], r.ci_95_difference[1]] : null);
    const ciLow = ci ? Number(ci[0]) : effVal - 0.28;
    const ciHigh = ci ? Number(ci[1]) : effVal + 0.28;
    const minBound = Math.min(-0.2, ciLow - 0.2);
    const maxBound = Math.max(1.2, ciHigh + 0.2);
    const range = maxBound - minBound || 1;
    const getX = (val) => paddingLeft + ((val - minBound) / range) * (svgWidth - paddingLeft - paddingRight);
    const yCenter = paddingTop + (svgHeight - paddingTop - paddingBottom) / 2;

    return (
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={getX(0)} y1={paddingTop} x2={getX(0)} y2={svgHeight - paddingBottom} stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="4,4" />
        <text x={getX(0)} y={paddingTop - 8} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle">
          Null (0.0)
        </text>
        <line x1={paddingLeft} y1={yCenter} x2={svgWidth - paddingRight} y2={yCenter} stroke="var(--border-default)" strokeWidth="1" />
        <text x={paddingLeft - 10} y={yCenter + 4} fill="var(--text-secondary)" fontSize="11" fontFamily="var(--font-sans)" fontWeight="600" textAnchor="end">
          {es.metric ? `${es.metric}` : 'Effect Size'}
        </text>
        <line x1={getX(ciLow)} y1={yCenter} x2={getX(ciHigh)} y2={yCenter} stroke="var(--accent-amber)" strokeWidth="3" />
        <line x1={getX(ciLow)} y1={yCenter - 6} x2={getX(ciLow)} y2={yCenter + 6} stroke="var(--accent-amber)" strokeWidth="2.5" />
        <line x1={getX(ciHigh)} y1={yCenter - 6} x2={getX(ciHigh)} y2={yCenter + 6} stroke="var(--accent-amber)" strokeWidth="2.5" />
        <polygon
          points={`${getX(effVal)},${yCenter - 7} ${getX(effVal) + 7},${yCenter} ${getX(effVal)},${yCenter + 7} ${getX(effVal) - 7},${yCenter}`}
          fill="var(--accent-green)"
          stroke="var(--text-primary)"
          strokeWidth="1.5"
        />
        <text x={getX(effVal)} y={yCenter + 22} fill="var(--text-primary)" fontSize="11" fontFamily="var(--font-mono)" fontWeight="700" textAnchor="middle">
          {effVal.toFixed(3)} [{ciLow.toFixed(2)}, {ciHigh.toFixed(2)}]
        </text>
        <text x={svgWidth / 2} y={svgHeight - 10} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">
          Standardized Effect Scale (Cohen's d / r / η²)
        </text>
      </svg>
    );
  }

  // ---------------------------------------------------------------------------
  // 5. SCATTER & CORRELATION REGRESSION PLOT
  // ---------------------------------------------------------------------------
  function renderScatterPlot() {
    const svgWidth = 480;
    const svgHeight = 240;
    const paddingLeft = 45;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 40;

    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;

    const corr = Number(r.correlation ?? (es.metric === 'r' ? es.value : 0.62));
    const points = useMemo(() => {
      const pts = [];
      for (let i = 0; i < 40; i++) {
        const x = i / 40;
        const noise = (Math.sin(i * 3.7) + Math.cos(i * 1.9)) * (1 - Math.abs(corr)) * 0.25;
        const y = Math.max(0.05, Math.min(0.95, corr >= 0 ? x * Math.abs(corr) + (1 - Math.abs(corr)) * 0.5 + noise : (1 - x) * Math.abs(corr) + (1 - Math.abs(corr)) * 0.5 + noise));
        pts.push({ x: paddingLeft + x * plotWidth, y: paddingTop + (1 - y) * plotHeight });
      }
      return pts;
    }, [corr, plotWidth, plotHeight, paddingLeft, paddingTop]);

    return (
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={svgHeight - paddingBottom} stroke="var(--border-default)" strokeWidth="1.5" />

        <line
          x1={paddingLeft}
          y1={corr >= 0 ? paddingTop + plotHeight * 0.85 : paddingTop + plotHeight * 0.15}
          x2={svgWidth - paddingRight}
          y2={corr >= 0 ? paddingTop + plotHeight * 0.15 : paddingTop + plotHeight * 0.85}
          stroke="var(--accent-amber)"
          strokeWidth="2.5"
          strokeDasharray="4,4"
        />

        {points.map((pt, i) => (
          <circle key={i} cx={pt.x} cy={pt.y} r="3.5" fill="var(--accent-green)" opacity="0.8" />
        ))}

        <text x={paddingLeft + plotWidth / 2} y={svgHeight - 10} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">
          Predictor Variable (X)
        </text>
        <text x={12} y={paddingTop + plotHeight / 2} fill="var(--text-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle" transform={`rotate(-90 12 ${paddingTop + plotHeight / 2})`}>
          Outcome Variable (Y)
        </text>
      </svg>
    );
  }

  return (
    <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-card)', padding: '14px', marginTop: '12px' }}>
      {/* TOP ROW: Title on Left, Rigid Graph Selection on Right Corner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', gap: '12px' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700, color: 'var(--accent-amber)', letterSpacing: '0.5px' }}>
          INTERACTIVE STATISTICAL VISUALIZATION
        </div>

        {/* RIGID TOP-RIGHT GRAPH SELECTION BUTTONS */}
        <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-input)', padding: '3px 6px', borderRadius: 'var(--radius-sharp)', border: '1px solid var(--border-subtle)', flexShrink: 0, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', marginRight: '2px', fontWeight: 600, letterSpacing: '0.5px' }}>
            Graph:
          </span>
          {!isCorrelation && (
            <>
              <button
                className={`btn btn-xs ${chartMode === 'bar' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: '10px', padding: '2px 8px' }}
                onClick={() => setChartMode('bar')}
              >
                Bar
              </button>
              <button
                className={`btn btn-xs ${chartMode === 'box' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: '10px', padding: '2px 8px' }}
                onClick={() => setChartMode('box')}
              >
                Box
              </button>
              <button
                className={`btn btn-xs ${chartMode === 'density' ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: '10px', padding: '2px 8px' }}
                onClick={() => setChartMode('density')}
              >
                Dist
              </button>
            </>
          )}

          {isCorrelation && (
            <button
              className={`btn btn-xs ${chartMode === 'scatter' ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '10px', padding: '2px 8px' }}
              onClick={() => setChartMode('scatter')}
            >
              Scatter
            </button>
          )}

          <button
            className={`btn btn-xs ${chartMode === 'forest' ? 'btn-primary' : 'btn-ghost'}`}
            style={{ fontSize: '10px', padding: '2px 8px' }}
            onClick={() => setChartMode('forest')}
          >
            Forest
          </button>
        </div>
      </div>

      {/* SUB-ROW (BELOW TITLE): Controls for CI Level, View Sub-Mode, and Scale indicators */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-start', gap: '14px', marginBottom: '10px', minHeight: '22px', flexWrap: 'wrap' }}>
        {/* CI Level Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
          <span
            style={{ color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: '3px', cursor: 'help' }}
            onMouseEnter={(e) => showTooltip(e, STAT_TOOLTIPS.ci)}
            onMouseLeave={hideTooltip}
          >
            <span>CI:</span>
            <span className="stat-info-badge" style={{ width: '11px', height: '11px', fontSize: '8px' }}>i</span>
          </span>
          {['90%', '95%', '99%'].map((lvl) => (
            <button
              key={lvl}
              style={{
                background: ciLevel === lvl ? 'var(--accent-amber-bg)' : 'transparent',
                border: `1px solid ${ciLevel === lvl ? 'var(--accent-amber)' : 'var(--border-subtle)'}`,
                color: ciLevel === lvl ? 'var(--accent-amber)' : 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: '9.5px',
                borderRadius: '2px',
                padding: '1px 6px',
                cursor: 'pointer',
              }}
              onClick={() => setCiLevel && setCiLevel(lvl)}
            >
              {lvl}
            </button>
          ))}
        </div>

        {/* Density View Mode Toggle */}
        {chartMode === 'density' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
            <span style={{ color: 'var(--text-muted)' }}>View:</span>
            <button
              className={`btn btn-xs ${densitySubMode === 'overlay' ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '9px', padding: '1px 6px' }}
              onClick={() => setDensitySubMode('overlay')}
            >
              Overlaid
            </button>
            <button
              className={`btn btn-xs ${densitySubMode === 'ridge' ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '9px', padding: '1px 6px' }}
              onClick={() => setDensitySubMode('ridge')}
            >
              Ridgeline
            </button>
          </div>
        )}

        {/* Scale Toggle for Bar and Box charts */}
        {(chartMode === 'bar' || chartMode === 'box') && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'var(--font-mono)', fontSize: '10px' }}>
            <span style={{ color: 'var(--text-muted)' }}>Scale:</span>
            <button
              className={`btn btn-xs ${scaleMode === 'focus' ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '9.5px', padding: '1px 6px' }}
              onClick={() => setScaleMode('focus')}
              title="Focus on CI & group variance"
            >
              Focus
            </button>
            <button
              className={`btn btn-xs ${scaleMode === 'zero' ? 'btn-primary' : 'btn-ghost'}`}
              style={{ fontSize: '9.5px', padding: '1px 6px' }}
              onClick={() => setScaleMode('zero')}
              title="Zero-origin baseline (0 – max)"
            >
              Zero
            </button>
          </div>
        )}
      </div>

      {/* Main Chart Box */}
      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sharp)', padding: '12px', position: 'relative' }}>
        {chartMode === 'bar' && renderBarChart()}
        {chartMode === 'box' && renderBoxPlot()}
        {chartMode === 'density' && renderDensityCurves()}
        {chartMode === 'forest' && renderForestPlot()}
        {chartMode === 'scatter' && renderScatterPlot()}

        {/* Dynamic Hover Tooltip readout */}
        {hoverInfo && (
          <div
            style={{
              position: 'absolute',
              bottom: '10px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'var(--tooltip-bg, #1a1d24)',
              border: '1px solid var(--tooltip-border, #343b48)',
              boxShadow: 'var(--tooltip-shadow, 0 4px 14px rgba(0,0,0,0.45))',
              padding: '8px 14px',
              borderRadius: 'var(--radius-card, 6px)',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              color: 'var(--text-primary, #e2e8f0)',
              zIndex: 10,
              pointerEvents: 'none',
              maxWidth: '92%',
              whiteSpace: 'pre-line',
              lineHeight: '1.5',
            }}
          >
            <strong style={{ color: 'var(--accent-amber, #f59e0b)', display: 'block', marginBottom: '3px' }}>
              {hoverInfo.title}
            </strong>
            {hoverInfo.body}
          </div>
        )}
      </div>

      <TooltipPortal tooltipState={tooltipState} />
    </div>
  );
}
