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

export function isTerminal(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed'
}
