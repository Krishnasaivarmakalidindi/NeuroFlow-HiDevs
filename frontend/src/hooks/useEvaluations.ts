import { useQuery } from '@tanstack/react-query';
import { evaluationService } from '../services/api';

export function useEvaluations(pipelineId?: string, threshold?: number, search?: string) {
  const evaluationsQuery = useQuery({
    queryKey: ['evaluations', pipelineId, threshold, search],
    queryFn: () => evaluationService.list(pipelineId, threshold, search),
  });

  return {
    evaluations: evaluationsQuery.data || [],
    isLoading: evaluationsQuery.isLoading,
    error: evaluationsQuery.error,
    refetch: evaluationsQuery.refetch,
  };
}
