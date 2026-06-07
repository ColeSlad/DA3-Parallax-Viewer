import { useEffect, useRef, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, TrackballControls } from '@react-three/drei'
import { DropInViewer } from '@mkkellogg/gaussian-splats-3d'
import * as THREE from 'three'
import type { ReconstructionResult } from '../api'

interface Props {
  result: ReconstructionResult
  onReset: () => void
}

export function SplatViewer({ result, onReset }: Props) {
  const [loaded, setLoaded] = useState(false)
  const [showControls, setShowControls] = useState(false)
  const savedCamera = useRef<{ _apply?: boolean } | null>(null)

  function handleReset() {
    if (savedCamera.current) savedCamera.current._apply = true
  }

  if (!result.splat_url) {
    return (
      <div style={containerStyle}>
        <p style={{ color: '#f87171', fontFamily: 'system-ui, sans-serif', textAlign: 'center', marginTop: 80 }}>
          No splat available for this reconstruction.
        </p>
        <button onClick={onReset} style={{ ...buttonStyle, display: 'block', margin: '16px auto' }}>Start over</button>
      </div>
    )
  }

  return (
    <div style={containerStyle}>
      {!loaded && <LoadingOverlay />}

      <Canvas camera={{ fov: 60, near: 0.01, far: 1000, up: [-1, 0, 0], position: [0, 0, 5] }}>
        <SplatScene
          splatUrl={result.splat_url}
          savedCamera={savedCamera}
          onLoaded={() => setLoaded(true)}
        />
        <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
          <GizmoViewport axisColors={['#f87171', '#4ade80', '#60a5fa']} labelColor="white" />
        </GizmoHelper>
      </Canvas>

      <div style={infoStyle}>
        <div style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>DA3-Parallax</div>
        <div>{result.view_count} views · {result.point_count.toLocaleString()} points</div>
        <div>{(result.duration_ms / 1000).toFixed(1)}s reconstruction</div>
      </div>

      <div style={{ position: 'absolute', top: 16, right: 16, display: 'flex', gap: 8, fontFamily: 'system-ui, sans-serif' }}>
        <button onClick={() => setShowControls(v => !v)} style={buttonStyle}>Controls</button>
        <button onClick={handleReset} style={buttonStyle}>Reset view</button>
        <button onClick={onReset} style={buttonStyle}>Start over</button>
      </div>

      {showControls && (
        <div style={controlsOverlayStyle}>
          <Row label="W / S" value="Move forward / backward" />
          <Row label="A / D" value="Strafe left / right" />
          <Row label="Q / E" value="Move up / down" />
          <Row label="R" value="Reset view" />
          <Row label="Left drag" value="Rotate" />
          <Row label="Right drag" value="Pan" />
          <Row label="Scroll" value="Zoom" />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------

interface SplatSceneProps {
  splatUrl: string
  savedCamera: React.MutableRefObject<{ _apply?: boolean } | null>
  onLoaded: () => void
}

function SplatScene({ splatUrl, savedCamera, onLoaded }: SplatSceneProps) {
  const { scene, camera } = useThree()
  const viewerRef = useRef<InstanceType<typeof DropInViewer> | null>(null)
  const keys = useRef(new Set<string>())
  const boundingRadius = useRef(1)
  const needsSave = useRef(false)

  useEffect(() => {
    const viewer = new DropInViewer({ gpuAcceleratedSort: true })
    scene.add(viewer)
    viewerRef.current = viewer

    // The splat PLY is in the DA3 world frame.
    // scale:[-1,1,-1] mirrors X and Z — the same transform the point cloud uses
    // (scale={[-1,1,-1]} on <points>) so both are consistent in screen space.
    // Canvas camera.up=[-1,0,0] (DA3 convention: +X is scene up).
    viewer.addSplatScene(splatUrl, {
      showLoadingUI: false,
      splatAlphaRemovalThreshold: 5,
      scale: [-1, 1, -1],
    }).then(() => {
      // Centre camera on the loaded scene
      if (viewer.splatMesh) {
        const box = new THREE.Box3().setFromObject(viewer.splatMesh)
        const sphere = new THREE.Sphere()
        box.getBoundingSphere(sphere)
        const r = sphere.radius || 1
        boundingRadius.current = r
        camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + r * 2.5)
        camera.near = r * 0.001
        camera.far = r * 20
        ;(camera as THREE.PerspectiveCamera).updateProjectionMatrix()
      }
      needsSave.current = true
      onLoaded()
    })

    return () => {
      viewer.dispose()
      scene.remove(viewer)
    }
    // intentionally omit onLoaded from deps — it's a stable callback from useState setter wrapper
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splatUrl, scene, camera])

  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT') return
      if (e.code === 'KeyR') {
        if (savedCamera.current) savedCamera.current._apply = true
        return
      }
      keys.current.add(e.code)
    }
    const onUp = (e: KeyboardEvent) => keys.current.delete(e.code)
    const clear = () => keys.current.clear()
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    window.addEventListener('blur', clear)
    document.addEventListener('visibilitychange', clear)
    return () => {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      window.removeEventListener('blur', clear)
      document.removeEventListener('visibilitychange', clear)
    }
  }, [savedCamera])

  const _fwd = new THREE.Vector3()
  const _right = new THREE.Vector3()
  const _move = new THREE.Vector3()

  useFrame((state, delta) => {
    const controls = state.controls as any

    if (needsSave.current && controls) {
      needsSave.current = false
      controls.saveState()
      savedCamera.current = { _apply: false }
    }

    if (savedCamera.current?._apply) {
      savedCamera.current._apply = false
      controls?.reset()
    }

    const k = keys.current
    if (k.size === 0) return
    const speed = boundingRadius.current * 2 * Math.min(delta, 0.05)
    camera.getWorldDirection(_fwd)
    _right.crossVectors(_fwd, camera.up).normalize()
    _move.set(0, 0, 0)
    if (k.has('KeyW')) _move.addScaledVector(_fwd, speed)
    if (k.has('KeyS')) _move.addScaledVector(_fwd, -speed)
    if (k.has('KeyA')) _move.addScaledVector(_right, -speed)
    if (k.has('KeyD')) _move.addScaledVector(_right, speed)
    if (k.has('KeyQ')) _move.addScaledVector(camera.up, speed)
    if (k.has('KeyE')) _move.addScaledVector(camera.up, -speed)
    if (_move.lengthSq() > 0) {
      camera.position.add(_move)
      controls?.target?.add(_move)
    }
  })

  return (
    <TrackballControls makeDefault rotateSpeed={3} zoomSpeed={1.2} panSpeed={0.8} staticMoving keys={['', '', '']} />
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

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 24 }}>
      <span style={{ color: '#f1f5f9', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#94a3b8' }}>{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------

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
}

const buttonStyle: React.CSSProperties = {
  padding: '8px 14px',
  background: '#1e293b',
  color: '#cbd5e1',
  border: '1px solid #334155',
  borderRadius: 6,
  fontSize: 13,
  cursor: 'pointer',
}

const controlsOverlayStyle: React.CSSProperties = {
  position: 'absolute',
  top: 56,
  right: 16,
  background: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 8,
  padding: '14px 18px',
  fontFamily: 'system-ui, sans-serif',
  fontSize: 13,
  color: '#cbd5e1',
  lineHeight: 2,
  minWidth: 220,
}
