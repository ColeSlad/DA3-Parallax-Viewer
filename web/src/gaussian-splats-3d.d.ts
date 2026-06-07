declare module '@mkkellogg/gaussian-splats-3d' {
  import * as THREE from 'three'

  interface SplatSceneOptions {
    showLoadingUI?: boolean
    splatAlphaRemovalThreshold?: number
    scale?: [number, number, number]
    position?: [number, number, number]
    rotation?: [number, number, number, number]
  }

  class DropInViewer extends THREE.Group {
    splatMesh: THREE.Object3D | null
    constructor(options?: { gpuAcceleratedSort?: boolean; sharedMemoryForWorkers?: boolean })
    addSplatScene(path: string, options?: SplatSceneOptions): Promise<void>
    addSplatScenes(scenes: Array<{ path: string } & SplatSceneOptions>, showLoadingUI?: boolean): Promise<void>
    removeSplatScene(index: number, showLoadingUI?: boolean): Promise<void>
    getSceneCount(): number
    dispose(): Promise<void>
  }

  export { DropInViewer }
}
