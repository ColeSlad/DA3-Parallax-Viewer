import { useQuery } from '@tanstack/react-query'
import { fetchJob, isTerminal, type Job } from '../api'

export function useJob(jobId: string | null) {
  return useQuery<Job>({
    queryKey: ['job', jobId],
    queryFn: () => fetchJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (!status || isTerminal(status)) return false
      return 1500
    },
  })
}
