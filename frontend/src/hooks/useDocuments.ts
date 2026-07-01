import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentService } from '../services/api';

export function useDocuments() {
  const queryClient = useQueryClient();

  const listQuery = useQuery({
    queryKey: ['documents'],
    queryFn: documentService.list,
  });

  const uploadMutation = useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress?: (e: any) => void }) => 
      documentService.upload(file, onProgress),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  return {
    documents: listQuery.data || [],
    isLoading: listQuery.isLoading,
    error: listQuery.error,
    refetch: listQuery.refetch,
    uploadDocument: uploadMutation.mutateAsync,
    isUploading: uploadMutation.isPending,
  };
}

export function useDocumentDetail(id: string | null) {
  const detailQuery = useQuery({
    queryKey: ['documents', id],
    queryFn: () => documentService.get(id!),
    enabled: !!id,
  });

  const similarQuery = useQuery({
    queryKey: ['documents', id, 'similar'],
    queryFn: () => documentService.similar(id!),
    enabled: !!id,
  });

  return {
    document: detailQuery.data || null,
    isLoading: detailQuery.isLoading,
    similarChunks: similarQuery.data || [],
    isLoadingSimilar: similarQuery.isLoading,
  };
}
