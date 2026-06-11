const BASE = import.meta.env.VITE_API_URL ?? ''

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface ReconstructionResult {
  pointcloud_url: string
  splat_url: string | null
  point_count: number
  view_count: number
  duration_ms: number
}

export interface InsertionResult {
  combined_splat_url: string
  duration_ms: number
}

export interface Job {
  job_id: string
  status: JobStatus
  created_at: string
  result: ReconstructionResult | InsertionResult | null
  error: string | null
}

export function submitImages(
  files: File[],
  onProgress?: (pct: number) => void,
): Promise<{ job_id: string; status: string }> {
  return new Promise((resolve, reject) => {
    const body = new FormData()
    for (const f of files) body.append('images', f)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/api/reconstructions`)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100))
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        reject(new Error(`Upload failed (${xhr.status}): ${xhr.responseText}`))
      }
    }

    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.send(body)
  })
}

export async function fetchJob(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/api/reconstructions/${jobId}`)
  if (!res.ok) throw new Error(`Poll failed (${res.status})`)
  return res.json()
}

export async function fetchInsertion(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/api/insertions/${jobId}`)
  if (!res.ok) throw new Error(`Poll failed (${res.status})`)
  return res.json()
}

export async function createInsertion(
  sceneJobId: string,
  body: { prompt: string; position: [number, number, number]; size_m: number; orientation?: string },
): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE}/api/scenes/${sceneJobId}/insertions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Insertion failed (${res.status}): ${await res.text()}`)
  return res.json()
}

export function isTerminal(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed'
}
