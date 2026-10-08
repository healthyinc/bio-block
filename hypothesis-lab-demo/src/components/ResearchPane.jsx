import QuestionCard from './QuestionCard';
import HypothesisCard from './HypothesisCard';
import ResultCard from './ResultCard';

export default function ResearchPane({
  profile,
  currentQuestion,
  hypotheses,
  analyses,
  validation,
  analysisResult,
  onAnswer,
  onRunAnalysis,
  answering,
}) {
  const analysesByHyp = {};
  (analyses || []).forEach((a) => {
    if (!analysesByHyp[a.hypothesis_id]) analysesByHyp[a.hypothesis_id] = [];
    analysesByHyp[a.hypothesis_id].push(a);
  });

  const hasContent = currentQuestion || (hypotheses && hypotheses.length > 0) || analysisResult;

  return (
    <div className="pane research-pane">
      <div className="pane-header">
        <h2>
          <span className="pane-header-num">03</span>
          HYPOTHESIS PROTOCOL
        </h2>
      </div>
      <div className="pane-body">
        {!hasContent ? (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '40px 0' }}>
            UPLOAD DATASET TO START HYPOTHESIS FORMULATION
          </div>
        ) : (
          <>
            {validation && validation.contradictions && validation.contradictions.length > 0 && (
              <div style={{ marginBottom: '12px', padding: '10px', background: 'var(--accent-red-bg)', border: '1px solid var(--accent-red)', color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                CONTRADICTION: {validation.contradictions.join('; ')}
              </div>
            )}

            {analysisResult && (
              <div style={{ marginBottom: '16px' }}>
                <div className="section-label">STATISTICAL TEST RESULT</div>
                <ResultCard result={analysisResult} onAnswer={onAnswer} />
              </div>
            )}

            {currentQuestion && (
              <QuestionCard
                question={currentQuestion}
                profile={profile}
                onAnswer={onAnswer}
                disabled={answering}
              />
            )}

            {hypotheses && hypotheses.length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <div className="section-label">CANDIDATE HYPOTHESES ({hypotheses.length})</div>
                {hypotheses.map((hyp) => (
                  <HypothesisCard
                    key={hyp.id}
                    hypothesis={hyp}
                    analyses={analysesByHyp[hyp.id] || []}
                    onRunAnalysis={onRunAnalysis}
                    analysisResult={analysisResult}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
