// attestation: verify sig -> ipfs -> on-chain

const BASE = '/demo/hypothesis-tree';

export async function submitAttestation(sessionId, {
  walletAddress,
  signature,
  timestamp,
  nonce,
  requestHash,
  sourceCid,
  branchId,
}) {
  const url = `${BASE}/sessions/${sessionId}/attest`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      wallet_address: walletAddress,
      signature,
      timestamp,
      nonce,
      request_hash: requestHash,
      source_cid: sourceCid || undefined,
      branch_id: branchId || undefined,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Attestation failed: ${res.status}`);
  }

  return res.json();
}

export async function getSessionAttestations(sessionId) {
  const url = `${BASE}/sessions/${sessionId}/attestations`;
  const res = await fetch(url);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch attestations: ${res.status}`);
  }

  return res.json();
}
