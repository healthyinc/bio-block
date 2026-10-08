import { useState } from 'react';
import { useTooltip, TooltipPortal } from './Tooltip';
import { FOLLOWUP_EXPLANATIONS } from './statDefinitions';

export default function QuestionCard({ question, profile, onAnswer, disabled }) {
  const [customText, setCustomText] = useState('');
  const { tooltipState, showTooltip, hideTooltip } = useTooltip();

  if (!question) return null;

  const displayExplanation = question.explanation;

  const isMeasurementQuestion =
    question.prompt?.toLowerCase().includes('independent') ||
    question.prompt?.toLowerCase().includes('paired') ||
    question.prompt?.toLowerCase().includes('repeated') ||
    question.prompt?.toLowerCase().includes('measurement');

  // auto-detect paired columns from names
  const autoDetectResult = (function () {
    if (!isMeasurementQuestion || !profile?.columns) return null;
    const cols = profile.columns.map((c) => c.name.toLowerCase().trim());
    const matchedPairs = [];

    for (let i = 0; i < cols.length; i++) {
      for (let j = i + 1; j < cols.length; j++) {
        const c1 = cols[i];
        const c2 = cols[j];
        if (
          (c1.includes('pre') && c2.includes('post')) ||
          (c1.includes('before') && c2.includes('after')) ||
          (c1.includes('baseline') && (c2.includes('followup') || c2.includes('follow_up'))) ||
          (c1.includes('time_1') && c2.includes('time_2'))
        ) {
          matchedPairs.push(`'${profile.columns[i].name}' & '${profile.columns[j].name}'`);
        }
      }
    }

    if (matchedPairs.length > 0) {
      return {
        recommendedType: 'paired',
        reason: `Auto-detected paired timepoints in your data (${matchedPairs[0]})`,
      };
    }

    return {
      recommendedType: 'independent',
      reason: 'No matching pre/post timepoint columns found; assuming independent cohorts',
    };
  })();

  function handleOptionClick(option) {
    if (option.disabled || disabled) return;
    onAnswer({ optionId: option.id, customAnswer: null });
  }

  function handleCustomSubmit() {
    if (!customText.trim() || disabled) return;
    onAnswer({ optionId: null, customAnswer: customText.trim() });
    setCustomText('');
  }

  return (
    <div className="glass-card question-card" style={{ border: '1.5px solid var(--border-default)' }}>
      <div className="question-prompt">{question.prompt}</div>

      {displayExplanation && (
        <div className="question-explanation" style={{ lineHeight: '1.55', color: 'var(--text-secondary)' }}>
          {displayExplanation}
        </div>
      )}

      {autoDetectResult && (
        <div style={{
          margin: '10px 0 14px 0',
          padding: '8px 12px',
          background: 'var(--accent-amber-bg)',
          border: '1px solid var(--accent-amber-border)',
          borderRadius: 'var(--radius-sharp)',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: 'var(--accent-amber)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span className="stat-info-badge" style={{ background: 'var(--accent-amber-bg)', color: 'var(--accent-amber)', borderColor: 'var(--accent-amber)' }}>i</span>
          <div>
            <strong>DATA-DRIVEN RECOMMENDATION:</strong>{' '}
            {autoDetectResult.reason}. We highlighted the recommended choice below.
          </div>
        </div>
      )}

      <div className="question-options">
        {question.options.map((opt) => {
          const exp = FOLLOWUP_EXPLANATIONS[opt.id];
          const hasTooltip = Boolean(exp);
          const optionDescription = opt.description || exp?.description;

          const optLower = (opt.id + ' ' + opt.label).toLowerCase();
          const isRecommended =
            autoDetectResult &&
            ((autoDetectResult.recommendedType === 'paired' && (optLower.includes('paired') || optLower.includes('repeated'))) ||
              (autoDetectResult.recommendedType === 'independent' && optLower.includes('independent')));

          return (
            <button
              key={opt.id}
              className={`question-option ${isRecommended ? 'suggested' : ''}`}
              onClick={() => handleOptionClick(opt)}
              disabled={opt.disabled || disabled}
              style={{
                borderColor: isRecommended ? 'var(--accent-amber)' : undefined,
                background: isRecommended ? 'var(--accent-amber-bg)' : undefined,
              }}
              onMouseEnter={(e) => {
                if (exp) {
                  showTooltip(e, { title: exp.title, body: exp.body, position: 'right' });
                }
              }}
              onMouseLeave={hideTooltip}
            >
              <div className="option-radio" />
              <div className="option-content" style={{ flex: 1 }}>
                <div className="option-label" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span>{opt.label}</span>
                  <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                    {isRecommended && (
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '9px',
                        fontWeight: 700,
                        color: 'var(--accent-amber)',
                        background: 'var(--bg-primary)',
                        border: '1px solid var(--accent-amber)',
                        padding: '1px 5px',
                        borderRadius: '2px',
                        letterSpacing: '0.5px',
                      }}>
                        RECOMMENDED
                      </span>
                    )}
                    {hasTooltip && (
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '10px',
                          color: 'var(--accent-amber)',
                          border: '1px solid var(--accent-amber)',
                          padding: '0 4px',
                          borderRadius: '2px',
                          cursor: 'help',
                        }}
                      >
                        INFO
                      </span>
                    )}
                  </div>
                </div>
                {optionDescription && (
                  <div className="option-description" style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px', lineHeight: '1.45' }}>
                    {optionDescription}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {question.allows_custom && (
        <div className="custom-input-group">
          <input
            type="text"
            className="custom-input"
            placeholder="Type custom response…"
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            disabled={disabled}
          />
          <button
            className="btn btn-primary"
            onClick={handleCustomSubmit}
            disabled={!customText.trim() || disabled}
          >
            SUBMIT
          </button>
        </div>
      )}

      <TooltipPortal tooltipState={tooltipState} />
    </div>
  );
}
