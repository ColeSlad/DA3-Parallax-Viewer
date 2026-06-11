declare module '@mkkellogg/gaussian-splats-3d' {
  import * as THREE from 'three'

  const SceneFormat: { Splat: 0; KSplat: 1; Ply: 2; Spz: 3 }

  interface SplatSceneOptions {
    showLoadingUI?: boolean
    splatAlphaRemovalThreshold?: number
    scale?: [number, number, number]
    position?: [number, number, number]
    rotation?: [number, number, number, number]
    format?: 0 | 1 | 2 | 3
  }

  interface ViewerOptions {
    rootElement?: HTMLElement | null
    selfDrivenMode?: boolean
    useBuiltInControls?: boolean
    gpuAcceleratedSort?: boolean
    sharedMemoryForWorkers?: boolean
    renderer?: THREE.WebGLRenderer
    camera?: THREE.Camera
    cameraUp?: [number, number, number]
    initialCameraPosition?: [number, number, number]
    initialCameraLookAt?: [number, number, number]
    dropInMode?: boolean
    logLevel?: number
  }

  class Viewer {
    splatMesh: THREE.Object3D | null
    constructor(options?: ViewerOptions)
    init(): void
    start(): void
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void>
    addSplatScenes(scenes: Array<{ path: string } & SplatSceneOptions>, showLoadingUI?: boolean): Promise<void>
    update(renderer?: THREE.WebGLRenderer, camera?: THREE.Camera): void
    render(): void
    dispose(): Promise<void>
  }

  class DropInViewer extends THREE.Group {
    viewer: Viewer
    splatMesh: THREE.Object3D | null
    constructor(options?: { gpuAcceleratedSort?: boolean; sharedMemoryForWorkers?: boolean })
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void>
    dispose(): Promise<void>
  }

  export { Viewer, DropInViewer, SceneFormat }
}
