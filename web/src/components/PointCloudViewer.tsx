import { Suspense, useEffect, useRef, useState, type CSSProperties } from 'react'
import { Canvas, useLoader, useThree, useFrame } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Html, TrackballControls } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'
import type { JobResult } from '../api'

interface Props {
  result: JobResult
  onReset: () => void
}

export function PointCloudViewer({ result, onReset }: Props) {
  const savedCamera = useRef<{ pos: THREE.Vector3; quat: THREE.Quaternion; _apply?: boolean } | null>(null)
  const [showControls, setShowControls] = useState(false)

  function handleReset() {
    if (savedCamera.current) savedCamera.current._apply = true
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0f172a' }}>
      <Canvas camera={{ fov: 60, near: 0.01, far: 1000, up: [-1, 0, 0], position: [0, 0, 5] }}>
        <Suspense fallback={<LoadingOverlay />}>
          <SceneContent url={result.ply_url} savedCamera={savedCamera} />
          <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
            <GizmoViewport
              axisColors={['#f87171', '#4ade80', '#60a5fa']}
              labelColor="white"
            />
          </GizmoHelper>
        </Suspense>
      </Canvas>

      {/* Info */}
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        color: '#94a3b8',
        fontFamily: 'system-ui, sans-serif',
        fontSize: 13,
        lineHeight: 1.6,
        pointerEvents: 'none',
      }}>
        <div style={{ color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>DA3-Parallax</div>
        <div>{result.view_count} views · {result.point_count.toLocaleString()} points</div>
        <div>{(result.duration_ms / 1000).toFixed(1)}s reconstruction</div>
      </div>

      {/* Buttons */}
      <div style={{
        position: 'absolute',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
        fontFamily: 'system-ui, sans-serif',
      }}>
        <button onClick={() => setShowControls(v => !v)} style={buttonStyle}>Controls</button>
        <button onClick={handleReset} style={buttonStyle}>Reset view</button>
        <button onClick={onReset} style={buttonStyle}>Start over</button>
      </div>

      {/* Controls overlay */}
      {showControls && (
        <div style={{
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
        }}>
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

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 24 }}>
      <span style={{ color: '#f1f5f9', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#94a3b8' }}>{value}</span>
    </div>
  )
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

interface SceneContentProps {
  url: string
  savedCamera: React.MutableRefObject<{ pos: THREE.Vector3; quat: THREE.Quaternion; _apply?: boolean } | null>
}

function SceneContent({ url, savedCamera }: SceneContentProps) {
  const geometry = useLoader(PLYLoader, url)
  const { camera } = useThree()
  const [offset, setOffset] = useState<[number, number, number]>([0, 0, 0])
  const boundingRadius = useRef(1)
  const keys = useRef(new Set<string>())
  const needsSave = useRef(false)

  useEffect(() => {
    geometry.computeBoundingSphere()
    const sphere = geometry.boundingSphere!
    const c = sphere.center
    setOffset([-c.x, -c.y, -c.z])

    const r = sphere.radius
    boundingRadius.current = r
    camera.position.set(0, 0, r * 2.5)
    camera.lookAt(0, 0, 0)
    camera.near = r * 0.001
    camera.far = r * 20
    camera.updateProjectionMatrix()

    // Signal the frame loop to saveState once controls are ready
    needsSave.current = true
  }, [geometry, camera])

  // WASD keyboard listeners
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
    const clearKeys = () => keys.current.clear()

    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    window.addEventListener('blur', clearKeys)
    document.addEventListener('visibilitychange', clearKeys)
    return () => {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      window.removeEventListener('blur', clearKeys)
      document.removeEventListener('visibilitychange', clearKeys)
    }
  }, [])

  const _forward = new THREE.Vector3()
  const _right = new THREE.Vector3()
  const _move = new THREE.Vector3()

  useFrame((state, delta) => {
    const controls = state.controls as any

    // Save initial state once controls are ready
    if (needsSave.current && controls) {
      needsSave.current = false
      controls.target.set(0, 0, 0)
      controls.saveState()
      savedCamera.current = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), _apply: false }
    }

    // Reset
    if (savedCamera.current?._apply) {
      savedCamera.current._apply = false
      controls?.reset()
    }

    // WASD / QE movement
    const k = keys.current
    if (k.size === 0) return

    const speed = boundingRadius.current * 2 * Math.min(delta, 0.05)
    camera.getWorldDirection(_forward)
    _right.crossVectors(_forward, camera.up).normalize()

    _move.set(0, 0, 0)
    if (k.has('KeyW')) _move.addScaledVector(_forward, speed)
    if (k.has('KeyS')) _move.addScaledVector(_forward, -speed)
    if (k.has('KeyA')) _move.addScaledVector(_right, -speed)
    if (k.has('KeyD')) _move.addScaledVector(_right, speed)
    if (k.has('KeyQ')) _move.addScaledVector(camera.up, speed)
    if (k.has('KeyE')) _move.addScaledVector(camera.up, -speed)

    if (_move.lengthSq() > 0) {
      camera.position.add(_move)
      controls?.target?.add(_move)
    }
  })

  const hasColors = !!geometry.attributes.color

  return (
    <>
      <points geometry={geometry} position={offset} scale={[-1, 1, -1]}>
        <pointsMaterial
          vertexColors={hasColors}
          color={hasColors ? undefined : '#7dd3fc'}
          size={0.005}
          sizeAttenuation
        />
      </points>
      {/* makeDefault registers controls in useThree so GizmoHelper can find them */}
      {/* keys={['','','']} prevents TrackballControls from intercepting A/S/D (its default pan/rotate/zoom keys), which caused double-movement during WASD+drag */}
      <TrackballControls makeDefault rotateSpeed={3} zoomSpeed={1.2} panSpeed={0.8} staticMoving keys={['', '', '']} />
    </>
  )
}

function LoadingOverlay() {
  return (
    <Html center>
      <div style={{ color: '#94a3b8', fontFamily: 'system-ui, sans-serif', fontSize: 14 }}>
        Loading point cloud…
      </div>
    </Html>
  )
}
