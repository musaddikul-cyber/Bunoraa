"use client";

import Link from "next/link";
import * as React from "react";
import Image from "next/image";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { getAccessToken } from "@/lib/auth";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type ChatUser = {
  id: string;
  email?: string;
  full_name?: string;
  avatar_url?: string | null;
};

type ChatAgent = {
  id: string;
  display_name?: string;
  avatar_url?: string | null;
  role?: string;
};

type ChatMessage = {
  id: string;
  content: string;
  is_from_customer: boolean;
  is_from_bot?: boolean;
  sender?: ChatUser | null;
  sender_display_name?: string;
  sender_avatar_url?: string | null;
  sender_role?: string;
  created_at: string;
};

type ChatConversation = {
  id: string;
  agent?: ChatAgent | null;
  messages?: ChatMessage[];
};

type ActiveConversationPayload =
  | ChatConversation
  | {
      conversation?: null;
      detail?: string;
    };

function normalizeActiveConversation(payload: ActiveConversationPayload | null | undefined) {
  if (!payload) return null;
  if ("id" in payload && payload.id) return payload as ChatConversation;
  return null;
}

function buildChatWsUrl(conversationId: string, token?: string | null) {
  const base = (process.env.NEXT_PUBLIC_WS_BASE_URL || "").replace(/\/$/, "");
  if (!base) return null;
  const path = `/ws/chat/${conversationId}/`;
  const normalizedPath = base.endsWith("/ws") ? path.replace(/^\/ws/, "") : path;
  const url = `${base}${normalizedPath}`;
  if (!token) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}

function initials(name?: string) {
  if (!name) return "?";
  const chunks = name
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2);
  if (!chunks.length) return "?";
  return chunks.map((part) => part[0]?.toUpperCase() || "").join("");
}

export function ChatWidget() {
  const queryClient = useQueryClient();
  const { hasToken } = useAuthContext();
  const wsEnabled = (process.env.NEXT_PUBLIC_WS_ENABLED || "").toLowerCase() === "true";
  const [open, setOpen] = React.useState(false);
  const [input, setInput] = React.useState("");
  const [wsState, setWsState] = React.useState<"idle" | "connecting" | "open" | "error">("idle");

  const activeConversation = useQuery({
    queryKey: ["chat", "active"],
    queryFn: async () => {
      const response = await apiFetch<ActiveConversationPayload>("/chat/conversations/active/");
      return normalizeActiveConversation(response.data);
    },
    enabled: open && hasToken,
    retry: false,
  });

  const conversationId = activeConversation.data?.id;

  const conversationDetail = useQuery({
    queryKey: ["chat", "conversation", conversationId],
    queryFn: async () => {
      const response = await apiFetch<ChatConversation>(`/chat/conversations/${conversationId}/`);
      return response.data;
    },
    enabled: open && hasToken && Boolean(conversationId),
  });

  const createConversation = useMutation({
    mutationFn: async (initialMessage: string) => {
      const response = await apiFetch<ChatConversation>("/chat/conversations/", {
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
      queryClient.invalidateQueries({ queryKey: ["chat", "active"] });
    },
  });

  React.useEffect(() => {
    if (!wsEnabled || !open || !conversationId || !hasToken) {
      setWsState("idle");
      return;
    }

    const token = getAccessToken();
    const url = buildChatWsUrl(conversationId, token);
    if (!url) {
      setWsState("error");
      return;
    }

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pingTimer: ReturnType<typeof setInterval> | null = null;
    let closedByEffect = false;
    let attempt = 0;

    const connect = () => {
      setWsState("connecting");
      socket = new WebSocket(url);

      socket.onopen = () => {
        attempt = 0;
        setWsState("open");
        if (pingTimer) clearInterval(pingTimer);
        pingTimer = setInterval(() => {
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "ping" }));
          }
        }, 25000);
      };

      socket.onmessage = () => {
        queryClient.invalidateQueries({ queryKey: ["chat", "conversation", conversationId] });
        queryClient.invalidateQueries({ queryKey: ["chat", "active"] });
      };

      socket.onerror = () => {
        setWsState("error");
      };

      socket.onclose = () => {
        if (closedByEffect) return;
        if (pingTimer) clearInterval(pingTimer);
        attempt += 1;
        const delay = Math.min(15000, 1000 * 2 ** Math.min(attempt, 5));
        reconnectTimer = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closedByEffect = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pingTimer) clearInterval(pingTimer);
      if (socket) socket.close();
    };
  }, [conversationId, hasToken, open, queryClient, wsEnabled]);

  const messages = conversationDetail.data?.messages || activeConversation.data?.messages || [];
  const assignedAgent = conversationDetail.data?.agent || activeConversation.data?.agent;

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !hasToken) return;
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
        <div className="chat-widget-mobile-closed-offset fixed bottom-4 right-4 z-50 sm:bottom-6 sm:right-6">
          <button
            type="button"
            className="rounded-full bg-primary px-4 py-2 text-sm text-white shadow-lg"
            onClick={() => setOpen(true)}
          >
            Chat
          </button>
        </div>
      ) : null}

      {open ? (
        <div className="chat-widget-mobile-open-offset fixed inset-x-3 bottom-3 z-50 sm:inset-x-auto sm:bottom-6 sm:right-6">
          <div className="chat-widget-mobile-panel flex w-full min-h-[22rem] max-h-[calc(100dvh-1.5rem)] flex-col rounded-2xl border border-border bg-card p-4 shadow-xl sm:w-96 sm:max-h-[38rem]">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <div className="relative h-8 w-8 overflow-hidden rounded-full bg-muted">
                  {assignedAgent?.avatar_url ? (
                    <Image
                      src={assignedAgent.avatar_url}
                      alt={assignedAgent.display_name || "Support"}
                      fill
                      sizes="32px"
                      unoptimized
                      className="object-cover"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-[11px] font-semibold text-foreground/60">
                      {initials(assignedAgent?.display_name || "Support")}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-sm font-semibold">Support chat</p>
                  <p className="text-[11px] text-foreground/60">
                    {wsState === "open" ? "Live" : wsState === "connecting" ? "Connecting..." : "Offline"}
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="rounded-full border border-border bg-background/80 px-2.5 py-1 text-xs text-foreground/70 shadow-sm transition hover:bg-muted hover:text-foreground"
                onClick={() => setOpen(false)}
                aria-label="Close chat"
              >
                Close
              </button>
            </div>

            {!hasToken ? (
              <div className="flex h-full flex-col items-start justify-center gap-3 rounded-xl border border-dashed border-border/70 bg-background/60 p-4">
                <p className="text-sm text-foreground/70">
                  Sign in to start a secure support chat with your account.
                </p>
                <Link
                  href="/account/login/"
                  className="rounded-full bg-primary px-4 py-2 text-xs font-semibold text-white"
                >
                  Sign in
                </Link>
              </div>
            ) : (
              <>
                <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1 text-sm">
                  {messages.length === 0 ? (
                    <p className="text-foreground/60">Start a conversation.</p>
                  ) : (
                    messages.map((msg) => {
                      const isMine = msg.is_from_customer;
                      const displayName = msg.sender_display_name || msg.sender?.full_name || "Support";
                      const avatarUrl = msg.sender_avatar_url || msg.sender?.avatar_url || null;
                      return (
                        <div
                          key={msg.id}
                          className={cn("flex gap-2", isMine ? "justify-end" : "justify-start")}
                        >
                          {!isMine ? (
                            <div className="relative h-8 w-8 shrink-0 overflow-hidden rounded-full bg-muted">
                              {avatarUrl ? (
                                <Image
                                  src={avatarUrl}
                                  alt={displayName}
                                  fill
                                  sizes="32px"
                                  unoptimized
                                  className="object-cover"
                                />
                              ) : (
                                <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-foreground/60">
                                  {initials(displayName)}
                                </div>
                              )}
                            </div>
                          ) : null}
                          <div
                            className={cn(
                              "max-w-[78%] rounded-2xl px-3 py-2",
                              isMine ? "bg-primary text-white" : "bg-muted"
                            )}
                          >
                            {!isMine ? (
                              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-foreground/60">
                                {displayName}
                              </p>
                            ) : null}
                            <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
                <div className="mt-3 flex gap-2">
                  <input
                    className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        handleSend();
                      }
                    }}
                    placeholder="Type a message"
                    disabled={sendMessage.isPending || createConversation.isPending}
                  />
                  <button
                    type="button"
                    className="rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white disabled:opacity-60"
                    onClick={handleSend}
                    disabled={sendMessage.isPending || createConversation.isPending}
                  >
                    Send
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </>
  );
}
