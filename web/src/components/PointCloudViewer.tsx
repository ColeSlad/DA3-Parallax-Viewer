import { Suspense, useEffect, useRef, useState } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Html } from '@react-three/drei'
import { TrackballControls } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'
import type { JobResult } from '../api'

interface Props {
  result: JobResult
  onReset: () => void
}

export function PointCloudViewer({ result, onReset }: Props) {
  const controlsRef = useRef<any>(null)

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0f172a', position: 'relative' }}>
      {/*
        Set up=[1,0,0] here so the camera is created with X as up BEFORE
        TrackballControls or the geometry useEffect runs.
      */}
      <Canvas camera={{ fov: 60, near: 0.01, far: 1000, up: [-1, 0, 0], position: [0, 0, 5] }}>
        <Suspense fallback={<LoadingOverlay />}>
          <SceneContent url={result.ply_url} controlsRef={controlsRef} />
          <GizmoHelper alignment="bottom-right" margin={[72, 72]}>
            <GizmoViewport
              axisColors={['#f87171', '#4ade80', '#60a5fa']}
              labelColor="white"
            />
          </GizmoHelper>
        </Suspense>
      </Canvas>

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
        <div style={{ marginTop: 8, fontSize: 11, color: '#475569' }}>
          Left drag · rotate &nbsp;·&nbsp; Right drag · pan &nbsp;·&nbsp; Scroll · zoom
        </div>
      </div>

      <div style={{
        position: 'absolute',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
        fontFamily: 'system-ui, sans-serif',
      }}>
        <button onClick={() => controlsRef.current?.reset()} style={buttonStyle}>
          Reset view
        </button>
        <button onClick={onReset} style={buttonStyle}>
          Start over
        </button>
      </div>
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
  controlsRef: React.RefObject<any>
}

function SceneContent({ url, controlsRef }: SceneContentProps) {
  const geometry = useLoader(PLYLoader, url)
  const { camera } = useThree()
  // Offset to center the cloud without mutating the cached geometry
  const [offset, setOffset] = useState<[number, number, number]>([0, 0, 0])

  useEffect(() => {
    geometry.computeBoundingSphere()
    const sphere = geometry.boundingSphere!
    const c = sphere.center

    // Center via mesh position, not geometry mutation — safe for useLoader cache
    setOffset([-c.x, -c.y, -c.z])

    const r = sphere.radius
    // Camera already has up=(-1,0,0) from Canvas prop; just set distance
    camera.position.set(0, 0, r * 2.5)
    camera.lookAt(0, 0, 0)
    camera.near = r * 0.001
    camera.far = r * 20
    camera.updateProjectionMatrix()

    setTimeout(() => controlsRef.current?.saveState(), 0)
  }, [geometry, camera, controlsRef])

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
      <TrackballControls
        ref={controlsRef}
        rotateSpeed={3}
        zoomSpeed={1.2}
        panSpeed={0.8}
      />
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
