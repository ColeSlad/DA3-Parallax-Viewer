const BASE = import.meta.env.VITE_API_URL ?? ''

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface JobResult {
  ply_url: string
  point_count: number
  view_count: number
  duration_ms: number
}

export interface Job {
  job_id: string
  status: JobStatus
  created_at: string
  result: JobResult | null
  error: string | null
}

export async function submitImages(files: File[]): Promise<{ job_id: string; status: string }> {
  const body = new FormData()
  for (const f of files) body.append('images', f)

  const res = await fetch(`${BASE}/api/reconstructions`, { method: 'POST', body })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Upload failed (${res.status}): ${text}`)
  }
  return res.json()
}

export async function fetchJob(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/api/reconstructions/${jobId}`)
  if (!res.ok) throw new Error(`Poll failed (${res.status})`)
  return res.json()
}

export function isTerminal(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed'
}
