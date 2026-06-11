import type { JobStatus } from '../api'

interface Props {
  status: JobStatus
  error: string | null
  onReset: () => void
}

export function StatusPanel({ status, error, onReset }: Props) {
  return (
    <div style={{
      maxWidth: 560,
      margin: '80px auto',
      fontFamily: 'system-ui, sans-serif',
      textAlign: 'center',
    }}>
      {(status === 'queued' || status === 'running' || status === 'succeeded') && (
        <>
          <Spinner />
          <p style={{ marginTop: 16, color: '#475569', fontSize: 16 }}>
            {status === 'queued' ? 'Queued — waiting for GPU…' : 'Processing…'}
          </p>
        </>
      )}

      {status === 'failed' && (
        <>
          <p style={{ color: '#dc2626', fontSize: 16 }}>Reconstruction failed.</p>
          {error && (
            <pre style={{
              background: '#fef2f2',
              border: '1px solid #fecaca',
              borderRadius: 6,
              padding: 12,
              fontSize: 12,
              textAlign: 'left',
              overflowX: 'auto',
              marginTop: 12,
            }}>
              {error}
            </pre>
          )}
          <ResetButton onReset={onReset} />
        </>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <div style={{
      width: 40,
      height: 40,
      border: '3px solid #e2e8f0',
      borderTop: '3px solid #2563eb',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
      margin: '0 auto',
    }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}

function ResetButton({ onReset }: { onReset: () => void }) {
  return (
    <button
      onClick={onReset}
      style={{
        marginTop: 20,
        padding: '10px 24px',
        background: '#2563eb',
        color: '#fff',
        border: 'none',
        borderRadius: 6,
        fontSize: 15,
        cursor: 'pointer',
      }}
    >
      Start over
    </button>
  )
}
