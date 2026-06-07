import { useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { submitImages, isTerminal, type ReconstructionResult } from './api'
import { useJob } from './hooks/useJob'
import { UploadForm } from './components/UploadForm'
import { StatusPanel } from './components/StatusPanel'
import { SplatViewer } from './components/SplatViewer'

const queryClient = new QueryClient()

type Phase = 'idle' | 'submitting' | 'reconstructing' | 'scene_ready' | 'error'

function Inner() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number>(0)

  const polling = phase === 'reconstructing' || phase === 'scene_ready' || phase === 'error'
  const { data: job } = useJob(polling ? jobId : null)

  useEffect(() => {
    if (phase === 'reconstructing' && job && isTerminal(job.status)) {
      setPhase(job.status === 'succeeded' ? 'scene_ready' : 'error')
    }
  }, [phase, job])

  async function handleSubmit(files: File[]) {
    setPhase('submitting')
    setSubmitError(null)
    setUploadProgress(0)
    try {
      const { job_id } = await submitImages(files, setUploadProgress)
      setJobId(job_id)
      setPhase('reconstructing')
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e))
      setPhase('idle')
    }
  }

  function reset() {
    setPhase('idle')
    setJobId(null)
    setSubmitError(null)
    queryClient.removeQueries({ queryKey: ['job'] })
  }

  if (phase === 'scene_ready' && job?.result) {
    return (
      <SplatViewer
        result={job.result as ReconstructionResult}
        onReset={reset}
      />
    )
  }

  if (phase === 'reconstructing' || phase === 'error') {
    const status = job?.status ?? 'queued'
    return (
      <StatusPanel
        status={phase === 'error' ? 'failed' : status}
        error={job?.error ?? null}
        onReset={reset}
      />
    )
  }

  return (
    <>
      <UploadForm
        onSubmit={handleSubmit}
        disabled={phase === 'submitting'}
        uploadProgress={phase === 'submitting' ? uploadProgress : null}
      />
      {submitError && (
        <p style={{
          maxWidth: 560,
          margin: '0 auto',
          color: '#dc2626',
          fontFamily: 'system-ui, sans-serif',
          fontSize: 14,
        }}>
          {submitError}
        </p>
      )}
    </>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Inner />
    </QueryClientProvider>
  )
}
