import { Suspense, useEffect, useRef } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'
import type { JobResult } from '../api'

interface Props {
  result: JobResult
  onReset: () => void
}

export function PointCloudViewer({ result, onReset }: Props) {
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0f172a', position: 'relative' }}>
      <Canvas camera={{ fov: 60, near: 0.01, far: 1000 }}>
        <Suspense fallback={<LoadingOverlay />}>
          <PointCloud url={result.ply_url} />
          <OrbitControls makeDefault />
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
      </div>

      <button
        onClick={onReset}
        style={{
          position: 'absolute',
          top: 16,
          right: 16,
          padding: '8px 16px',
          background: '#1e293b',
          color: '#cbd5e1',
          border: '1px solid #334155',
          borderRadius: 6,
          fontSize: 13,
          cursor: 'pointer',
        }}
      >
        Start over
      </button>
    </div>
  )
}

function PointCloud({ url }: { url: string }) {
  const geometry = useLoader(PLYLoader, url)
  const pointsRef = useRef<THREE.Points>(null)
  const { camera, controls } = useThree()

  useEffect(() => {
    if (!geometry) return

    geometry.computeBoundingSphere()
    const sphere = geometry.boundingSphere!

    // Center geometry at origin so OrbitControls target works naturally
    geometry.translate(-sphere.center.x, -sphere.center.y, -sphere.center.z)
    geometry.computeBoundingSphere()

    const r = geometry.boundingSphere!.radius
    const distance = r * 2.5

    camera.position.set(0, r * 0.5, distance)
    camera.near = r * 0.001
    camera.far = r * 20
    camera.updateProjectionMatrix()

    if (controls) {
      // @ts-expect-error – OrbitControls target
      controls.target.set(0, 0, 0)
      // @ts-expect-error
      controls.update()
    }
  }, [geometry, camera, controls])

  // PLYLoader returns Float32BufferAttribute colors in 0..1 range — no remapping needed.
  // If colors attribute is missing (grayscale PLY), fall back to a flat color.
  const hasColors = !!geometry.attributes.color

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        vertexColors={hasColors}
        color={hasColors ? undefined : '#7dd3fc'}
        size={0.005}
        sizeAttenuation
      />
    </points>
  )
}

function LoadingOverlay() {
  return (
    <Html center>
      <div style={{
        color: '#94a3b8',
        fontFamily: 'system-ui, sans-serif',
        fontSize: 14,
      }}>
        Loading point cloud…
      </div>
    </Html>
  )
}
