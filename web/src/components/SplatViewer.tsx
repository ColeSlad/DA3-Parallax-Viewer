import { useEffect, useRef, useState } from 'react'
import { Viewer, SceneFormat } from '@mkkellogg/gaussian-splats-3d'

interface Props {
  splatUrl: string
  onReset: () => void
}

export function SplatViewer({ splatUrl, onReset }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (!containerRef.current) return

    let active = true

    const viewer = new Viewer({
      rootElement: containerRef.current,
      selfDrivenMode: true,
      useBuiltInControls: true,
      gpuAcceleratedSort: false,
      sharedMemoryForWorkers: false,
    })

    viewer.addSplatScene(splatUrl, {
      format: SceneFormat.Ply,
      showLoadingUI: false,
      splatAlphaRemovalThreshold: 5,
    }).then(() => {
      if (!active) return
      viewer.start()
      setLoaded(true)
    }).catch((e: unknown) => {
      if (active) console.error('addSplatScene failed:', e)
    })

    return () => {
      active = false
      viewer.dispose().catch(() => {})
    }
  }, [splatUrl])

  return (
    <div style={containerStyle}>
      <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />

      {!loaded && <LoadingOverlay />}

      {loaded && (
        <div style={{ position: 'absolute', top: 16, right: 16, fontFamily: 'system-ui, sans-serif' }}>
          <button onClick={onReset} style={buttonStyle}>Start over</button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

function LoadingOverlay() {
  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0f172a',
      zIndex: 10,
      color: '#94a3b8',
      fontFamily: 'system-ui, sans-serif',
      fontSize: 14,
      gap: 16,
    }}>
      <Spinner />
      <span>Loading splat scene…</span>
      <span style={{ fontSize: 12, color: '#475569' }}>Gaussian splat files are large — this may take a moment</span>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{
      width: 36,
      height: 36,
      border: '3px solid #1e293b',
      borderTop: '3px solid #60a5fa',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}

const containerStyle: React.CSSProperties = { position: 'fixed', inset: 0, background: '#0f172a' }

const infoStyle: React.CSSProperties = {
  position: 'absolute',
  top: 16,
  left: 16,
  color: '#94a3b8',
  fontFamily: 'system-ui, sans-serif',
  fontSize: 13,
  lineHeight: 1.6,
  pointerEvents: 'none',
  zIndex: 1,
}

const buttonStyle: React.CSSProperties = {
  padding: '8px 14px',
  background: '#1e293b',
  color: '#cbd5e1',
  border: '1px solid #334155',
  borderRadius: 6,
  fontSize: 13,
  cursor: 'pointer',
  zIndex: 1,
  position: 'relative',
}
