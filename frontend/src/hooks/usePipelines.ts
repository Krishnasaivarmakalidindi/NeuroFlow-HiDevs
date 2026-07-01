import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { pipelineService } from '../services/api';
import { PipelineConfig } from '../types';

export function usePipelines() {
  const queryClient = useQueryClient();

  const listQuery = useQuery({
    queryKey: ['pipelines'],
    queryFn: pipelineService.list,
  });

  const createMutation = useMutation({
    mutationFn: pipelineService.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelines'] });
    },
  });

  return {
    pipelines: listQuery.data || [],
    isLoading: listQuery.isLoading,
    error: listQuery.error,
    refetch: listQuery.refetch,
    createPipeline: createMutation.mutateAsync,
    isCreating: createMutation.isPending,
  };
}

export function usePipelineDetail(id: string | null) {
  const queryClient = useQueryClient();

  const detailQuery = useQuery({
    queryKey: ['pipelines', id],
    queryFn: () => pipelineService.get(id!),
    enabled: !!id,
  });

  const updateMutation = useMutation({
    mutationFn: (config: PipelineConfig) => pipelineService.update(id!, config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelines'] });
      queryClient.invalidateQueries({ queryKey: ['pipelines', id] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => pipelineService.delete(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelines'] });
    },
  });

  return {
    pipeline: detailQuery.data || null,
    isLoading: detailQuery.isLoading,
    error: detailQuery.error,
    updatePipeline: updateMutation.mutateAsync,
    isUpdating: updateMutation.isPending,
    deletePipeline: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
  };
}

export function usePipelineAnalytics(id: string | null) {
  const analyticsQuery = useQuery({
    queryKey: ['pipelines', id, 'analytics'],
    queryFn: () => pipelineService.analytics(id!),
    enabled: !!id,
  });

  return {
    analytics: analyticsQuery.data || null,
    isLoading: analyticsQuery.isLoading,
    error: analyticsQuery.error,
  };
}

export function usePipelineRuns(id: string | null, page = 1, limit = 10) {
  const runsQuery = useQuery({
    queryKey: ['pipelines', id, 'runs', page, limit],
    queryFn: () => pipelineService.runs(id!, page, limit),
    enabled: !!id,
  });

  return {
    runs: runsQuery.data || [],
    isLoading: runsQuery.isLoading,
    error: runsQuery.error,
  };
}
