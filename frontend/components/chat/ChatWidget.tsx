"use client";

import * as React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type Conversation = {
  id: string;
  messages?: Array<{ id: string; content: string; is_from_customer: boolean; created_at: string }>;
};

function buildChatWsUrl(conversationId: string) {
  const base = (process.env.NEXT_PUBLIC_WS_BASE_URL || "").replace(/\/$/, "");
  if (!base) return null;
  const path = `/ws/chat/${conversationId}/`;
  const normalizedPath = base.endsWith("/ws") ? path.replace(/^\/ws/, "") : path;
  return `${base}${normalizedPath}`;
}

export function ChatWidget() {
  const [open, setOpen] = React.useState(false);
  const [input, setInput] = React.useState("");
  const queryClient = useQueryClient();

  const activeConversation = useQuery({
    queryKey: ["chat", "active"],
    queryFn: async () => {
      const response = await apiFetch<Conversation>("/chat/conversations/active/");
      return response.data;
    },
    enabled: open,
    retry: false,
  });

  const conversationId = activeConversation.data?.id;

  const conversationDetail = useQuery({
    queryKey: ["chat", "conversation", conversationId],
    queryFn: async () => {
      const response = await apiFetch<Conversation>(`/chat/conversations/${conversationId}/`);
      return response.data;
    },
    enabled: Boolean(conversationId),
  });

  const createConversation = useMutation({
    mutationFn: async (initialMessage: string) => {
      const response = await apiFetch<Conversation>("/chat/conversations/", {
        method: "POST",
        body: {
          category: "general",
          subject: "Support",
          initial_message: initialMessage,
          source: "website",
        },
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", "active"] });
    },
  });

  const sendMessage = useMutation({
    mutationFn: async (payload: { conversation: string; content: string }) => {
      return apiFetch("/chat/messages/", {
        method: "POST",
        body: { conversation: payload.conversation, content: payload.content, message_type: "text" },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat", "conversation", conversationId] });
    },
  });

  React.useEffect(() => {
    if (!conversationId) return;
    const url = buildChatWsUrl(conversationId);
    if (!url) return;
    const socket = new WebSocket(url);
    socket.onmessage = () => {
      queryClient.invalidateQueries({ queryKey: ["chat", "conversation", conversationId] });
    };
    return () => socket.close();
  }, [conversationId, queryClient]);

  const messages = conversationDetail.data?.messages || activeConversation.data?.messages || [];

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");

    if (conversationId) {
      await sendMessage.mutateAsync({ conversation: conversationId, content: text });
      return;
    }

    await createConversation.mutateAsync(text);
  };

  return (
    <>
      {!open ? (
        <div className="fixed bottom-4 right-4 z-50 sm:bottom-6 sm:right-6">
          <button
            className="rounded-full bg-primary px-4 py-2 text-sm text-white shadow-lg"
            onClick={() => setOpen(true)}
          >
            Chat
          </button>
        </div>
      ) : null}

      {open ? (
        <div className="fixed inset-x-3 bottom-3 z-50 sm:inset-x-auto sm:bottom-6 sm:right-6">
          <div className="flex w-full max-h-[calc(100dvh-1.5rem)] min-h-[18rem] flex-col rounded-2xl border border-border bg-card p-4 shadow-xl sm:w-80 sm:max-h-[34rem]">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-semibold">Support chat</div>
              <button
                type="button"
                className="rounded-full border border-border bg-background/80 px-2.5 py-1 text-xs text-foreground/70 shadow-sm transition hover:bg-muted hover:text-foreground"
                onClick={() => setOpen(false)}
                aria-label="Close chat"
              >
                Close
              </button>
            </div>
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1 text-sm">
              {messages.length === 0 ? (
                <p className="text-foreground/60">Start a conversation.</p>
              ) : (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={cn(
                      "rounded-lg px-3 py-2",
                      msg.is_from_customer ? "bg-primary text-white" : "bg-muted"
                    )}
                  >
                    {msg.content}
                  </div>
                ))
              )}
            </div>
            <div className="mt-3 flex gap-2">
              <input
                className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Type a message"
              />
              <button className="text-sm text-primary" onClick={handleSend}>
                Send
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
