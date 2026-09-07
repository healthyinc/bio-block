import { useState, useRef } from 'react';

export default function UploadScreen({ onUpload }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      await onUpload(file);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }

  function handleChange(e) {
    const file = e.target.files[0];
    handleFile(file);
  }

  if (loading) {
    return (
      <div className="upload-screen">
        <div className="upload-card">
          <div className="upload-loading">
            <div className="spinner"></div>
            <p style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: '13px' }}>
              PROFILING DATASET STRUCTURE & VARIABLES...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="upload-screen">
      <div className="upload-card">
        <div className="upload-tagline">BIO-BLOCK SCIENTIFIC WORKBENCH</div>
        <h1>Hypothesis Exploration & Provenance Engine</h1>
        <p>
          Upload a research dataset (.csv, .xlsx) to initialize automated variable profiling,
          interactive decision tree branching, and verifiable on-chain attestation.
        </p>

        <div
          className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleChange}
          />
          <div className="upload-icon-box">SELECT FILE</div>
          <div className="upload-text">
            Drop raw data file here or <strong>browse local filesystem</strong>
          </div>
          <div className="upload-hint">
            CSV, XLSX, XLS · Max 10,000 observations
          </div>
        </div>

        {error && (
          <div style={{ marginTop: '16px', padding: '10px', background: 'var(--accent-red-bg)', border: '1px solid var(--accent-red)', color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            ERROR: {error}
          </div>
        )}
      </div>
    </div>
  );
}
