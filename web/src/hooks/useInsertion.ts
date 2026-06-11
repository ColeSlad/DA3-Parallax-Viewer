import { useQuery } from '@tanstack/react-query'
import { fetchInsertion, isTerminal, type Job } from '../api'

export function useInsertion(jobId: string | null) {
  return useQuery<Job>({
    queryKey: ['insertion', jobId],
    queryFn: () => fetchInsertion(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (!status || isTerminal(status)) return false
      return 1500
    },
  })
}
