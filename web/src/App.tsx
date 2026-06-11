import { useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { submitImages, createInsertion, isTerminal, type ReconstructionResult, type InsertionResult } from './api'
import { useJob } from './hooks/useJob'
import { useInsertion } from './hooks/useInsertion'
import { UploadForm } from './components/UploadForm'
import { StatusPanel } from './components/StatusPanel'
import { PointCloudViewer } from './components/PointCloudViewer'
import { SplatViewer } from './components/SplatViewer'

const queryClient = new QueryClient()

type Phase = 'idle' | 'submitting' | 'reconstructing' | 'scene_ready' | 'inserting' | 'splat_ready' | 'error'

function Inner() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [insertionJobId, setInsertionJobId] = useState<string | null>(null)
  const [combinedSplatUrl, setCombinedSplatUrl] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number>(0)

  const polling = phase === 'reconstructing' || phase === 'scene_ready' || phase === 'error'
  const { data: job } = useJob(polling ? jobId : null)
  const { data: insertionJob } = useInsertion(phase === 'inserting' ? insertionJobId : null)

  useEffect(() => {
    if (phase === 'reconstructing' && job && isTerminal(job.status)) {
      setPhase(job.status === 'succeeded' ? 'scene_ready' : 'error')
    }
  }, [phase, job])

  useEffect(() => {
    if (phase === 'inserting' && insertionJob && isTerminal(insertionJob.status)) {
      if (insertionJob.status === 'succeeded') {
        const url = (insertionJob.result as InsertionResult | null)?.combined_splat_url
        if (url) {
          setCombinedSplatUrl(url)
          setPhase('splat_ready')
        } else {
          setPhase('error')
        }
      } else {
        setPhase('error')
      }
    }
  }, [phase, insertionJob])

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

  async function handleInsert(prompt: string, position: [number, number, number], size_m: number) {
    if (!jobId) return
    try {
      const { job_id } = await createInsertion(jobId, { prompt, position, size_m })
      setInsertionJobId(job_id)
      setPhase('inserting')
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e))
    }
  }

  function reset() {
    setPhase('idle')
    setJobId(null)
    setInsertionJobId(null)
    setCombinedSplatUrl(null)
    setSubmitError(null)
    queryClient.removeQueries({ queryKey: ['job'] })
    queryClient.removeQueries({ queryKey: ['insertion'] })
  }

  if (phase === 'splat_ready' && combinedSplatUrl) {
    return <SplatViewer splatUrl={combinedSplatUrl} onReset={reset} />
  }

  if (phase === 'inserting') {
    return (
      <StatusPanel
        status={insertionJob?.status ?? 'queued'}
        error={insertionJob?.error ?? null}
        onReset={reset}
      />
    )
  }

  if (phase === 'scene_ready' && job?.result) {
    return (
      <PointCloudViewer
        result={job.result as ReconstructionResult}
        onReset={reset}
        onInsert={handleInsert}
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
