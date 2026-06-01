import { useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { submitImages, isTerminal } from './api'
import { useJob } from './hooks/useJob'
import { UploadForm } from './components/UploadForm'
import { StatusPanel } from './components/StatusPanel'
import { PointCloudViewer } from './components/PointCloudViewer'

const queryClient = new QueryClient()

type Phase = 'idle' | 'submitting' | 'polling' | 'done' | 'error'

function Inner() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const { data: job } = useJob(phase === 'polling' || phase === 'done' || phase === 'error' ? jobId : null)

  // Advance phase when job reaches a terminal state
  useEffect(() => {
    if (phase === 'polling' && job && isTerminal(job.status)) {
      setPhase(job.status === 'succeeded' ? 'done' : 'error')
    }
  }, [phase, job])

  async function handleSubmit(files: File[]) {
    setPhase('submitting')
    setSubmitError(null)
    try {
      const { job_id } = await submitImages(files)
      setJobId(job_id)
      setPhase('polling')
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

  if (phase === 'done' && job?.result) {
    return <PointCloudViewer result={job.result} onReset={reset} />
  }

  if (phase === 'polling' || phase === 'error') {
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
      <UploadForm onSubmit={handleSubmit} disabled={phase === 'submitting'} />
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
