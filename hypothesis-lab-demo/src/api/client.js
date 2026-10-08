// proxy requests -> localhost:3003

const BASE = '/demo/hypothesis-tree';

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function createSession(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/sessions', { method: 'POST', body: form });
}

export async function getSession(sessionId, activeBranchId) {
  const query = activeBranchId ? `?active_branch_id=${activeBranchId}` : '';
  return request(`/sessions/${sessionId}${query}`);
}

export async function submitAnswer(sessionId, { parentNodeId, optionId, customAnswer, selectedColumns }) {
  return request(`/sessions/${sessionId}/answers`, {
    method: 'POST',
    body: JSON.stringify({
      parent_node_id: parentNodeId,
      option_id: optionId || undefined,
      custom_answer: customAnswer || undefined,
      selected_columns: selectedColumns || undefined,
    }),
  });
}

export async function forkBranch(sessionId, nodeId, newAnswer) {
  return request(`/sessions/${sessionId}/forks`, {
    method: 'POST',
    body: JSON.stringify({
      node_id: nodeId,
      new_answer: newAnswer || undefined,
    }),
  });
}

export async function manageHypothesis(sessionId, { action, hypothesisId, statement, annotation, branchId }) {
  return request(`/sessions/${sessionId}/hypotheses`, {
    method: 'POST',
    body: JSON.stringify({
      action,
      hypothesis_id: hypothesisId || undefined,
      statement: statement || undefined,
      annotation: annotation || undefined,
      branch_id: branchId || undefined,
    }),
  });
}

export async function runAnalysis(sessionId, { hypothesisId, analysisId }) {
  return request(`/sessions/${sessionId}/analyses`, {
    method: 'POST',
    body: JSON.stringify({
      hypothesis_id: hypothesisId,
      analysis_id: analysisId,
    }),
  });
}

export async function deleteSession(sessionId) {
  return request(`/sessions/${sessionId}`, { method: 'DELETE' });
}
