import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { DataExportJob } from "@/lib/types";

const exportKey = ["account", "exports"] as const;

async function fetchExports() {
  const response = await apiFetch<DataExportJob[]>("/accounts/export/");
  return response.data;
}

export function useExports() {
  const queryClient = useQueryClient();

  const exportsQuery = useQuery({
    queryKey: exportKey,
    queryFn: fetchExports,
  });

  const requestExport = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<DataExportJob>("/accounts/export/", {
        method: "POST",
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: exportKey });
    },
  });

  return {
    exportsQuery,
    requestExport,
  };
}
