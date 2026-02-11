import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  PreorderCategory,
  PreorderTemplate,
  Preorder,
  PreorderPriceEstimate,
  PreorderDesign,
  PreorderReference,
  PreorderMessage,
  PreorderOptionValue,
  PreorderStatistics,
} from "@/lib/types";

export function usePreorderCategories() {
  return useQuery({
    queryKey: ["preorders", "categories"],
    queryFn: async () => {
      const response = await apiFetch<PreorderCategory[]>("/preorders/categories/");
      return response.data;
    },
  });
}

export function usePreorderCategory(slug?: string | null) {
  return useQuery({
    queryKey: ["preorders", "category", slug],
    queryFn: async () => {
      const response = await apiFetch<PreorderCategory>(`/preorders/categories/${slug}/`);
      return response.data;
    },
    enabled: Boolean(slug),
  });
}

export function usePreorderTemplates(categorySlug?: string | null) {
  return useQuery({
    queryKey: ["preorders", "templates", categorySlug],
    queryFn: async () => {
      const response = await apiFetch<PreorderTemplate[]>("/preorders/templates/", {
        params: {
          category: categorySlug || undefined,
        },
      });
      return response.data;
    },
    enabled: Boolean(categorySlug),
  });
}

export function usePreorderEstimate() {
  return useMutation({
    mutationFn: async (payload: {
      category_id: string;
      quantity: number;
      options?: Record<string, unknown>;
      is_rush_order?: boolean;
    }) => {
      const response = await apiFetch<PreorderPriceEstimate>("/preorders/calculate-price/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
      return response.data;
    },
  });
}

export function useCreatePreorder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      const response = await apiFetch<Preorder>("/preorders/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["preorders", "list"] });
    },
  });
}

export function usePreorderTracking() {
  return useMutation({
    mutationFn: async (payload: { preorder_number: string; email: string }) => {
      const response = await apiFetch<Preorder>("/preorders/track/", {
        method: "POST",
        body: payload,
        allowGuest: true,
      });
      return response.data;
    },
  });
}

export function usePreorderDetail(preorderNumber?: string | null) {
  return useQuery({
    queryKey: ["preorders", preorderNumber],
    queryFn: async () => {
      const response = await apiFetch<Preorder>(
        `/preorders/${preorderNumber}/`
      );
      return response.data;
    },
    enabled: Boolean(preorderNumber),
  });
}

export function usePreorderActions(preorderNumber?: string | null) {
  const queryClient = useQueryClient();
  const invalidate = () => {
    if (preorderNumber) {
      queryClient.invalidateQueries({ queryKey: ["preorders", preorderNumber] });
    }
    queryClient.invalidateQueries({ queryKey: ["preorders", "list"] });
  };

  const sendMessage = useMutation({
    mutationFn: async (payload: { message: string; subject?: string }) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const response = await apiFetch<PreorderMessage>(
        `/preorders/${preorderNumber}/send_message/`,
        { method: "POST", body: payload }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  const uploadDesign = useMutation({
    mutationFn: async (payload: { file: File; notes?: string }) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const form = new FormData();
      form.append("file", payload.file);
      if (payload.notes) form.append("notes", payload.notes);
      const response = await apiFetch<PreorderDesign>(
        `/preorders/${preorderNumber}/upload_design/`,
        {
          method: "POST",
          body: form,
        }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  const uploadReference = useMutation({
    mutationFn: async (payload: { file: File; description?: string }) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const form = new FormData();
      form.append("file", payload.file);
      if (payload.description) form.append("description", payload.description);
      const response = await apiFetch<PreorderReference>(
        `/preorders/${preorderNumber}/upload_reference/`,
        {
          method: "POST",
          body: form,
        }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  const uploadOptionFile = useMutation({
    mutationFn: async (payload: { optionId: string; file: File }) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const form = new FormData();
      form.append("option_id", payload.optionId);
      form.append("file", payload.file);
      const response = await apiFetch<PreorderOptionValue>(
        `/preorders/${preorderNumber}/upload-option-file/`,
        {
          method: "POST",
          body: form,
        }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  const acceptQuote = useMutation({
    mutationFn: async (quoteId: string) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const response = await apiFetch<Preorder>(
        `/preorders/${preorderNumber}/accept_quote/`,
        { method: "POST", body: { quote_id: quoteId } }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  const rejectQuote = useMutation({
    mutationFn: async (payload: { quoteId: string; reason?: string }) => {
      if (!preorderNumber) throw new Error("Missing preorder number");
      const response = await apiFetch<Preorder>(
        `/preorders/${preorderNumber}/reject_quote/`,
        {
          method: "POST",
          body: { quote_id: payload.quoteId, reason: payload.reason || "" },
        }
      );
      return response.data;
    },
    onSuccess: invalidate,
  });

  return {
    sendMessage,
    uploadDesign,
    uploadReference,
    uploadOptionFile,
    acceptQuote,
    rejectQuote,
  };
}

export function usePreorderStatistics() {
  return useQuery({
    queryKey: ["preorders", "stats"],
    queryFn: async () => {
      const response = await apiFetch<PreorderStatistics>("/preorders/statistics/");
      return response.data;
    },
  });
}
