"use client";

import Link from "next/link";
import * as React from "react";
import Image from "next/image";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Paperclip, SendHorizontal, X } from "lucide-react";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { getAccessToken } from "@/lib/auth";
import { apiFetch } from "@/lib/api";
import { buildWsUrl } from "@/lib/ws";
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

type ChatAttachment = {
  id: string;
  file?: string | null;
  file_name?: string;
  file_type?: string;
  file_size?: number;
  download_url?: string | null;
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
  attachments?: ChatAttachment[];
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
  return buildWsUrl(`/ws/chat/${conversationId}/`, token);
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

function formatFileSize(size?: number) {
  if (!size || size <= 0) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function ChatWidget() {
  const queryClient = useQueryClient();
  const { hasToken } = useAuthContext();
  const wsEnabled = (process.env.NEXT_PUBLIC_WS_ENABLED || "").toLowerCase() === "true";
  const [open, setOpen] = React.useState(false);
  const [input, setInput] = React.useState("");
  const [pendingFiles, setPendingFiles] = React.useState<File[]>([]);
  const [wsState, setWsState] = React.useState<"idle" | "connecting" | "open" | "error">("idle");
  const autoGreetingSentRef = React.useRef(false);
  const composerRef = React.useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  const resizeComposer = React.useCallback(() => {
    const textarea = composerRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    const computedStyles = window.getComputedStyle(textarea);
    const lineHeight = Number.parseFloat(computedStyles.lineHeight) || 20;
    const minHeight = 40;
    const maxHeight = Math.round(lineHeight * 5 + 20);
    const nextHeight = Math.max(minHeight, Math.min(textarea.scrollHeight, maxHeight));

    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, []);

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
    onError: () => {
      autoGreetingSentRef.current = false;
    },
  });

  const sendMessage = useMutation({
    mutationFn: async (payload: { conversation: string; content: string; files?: File[] }) => {
      const files = payload.files || [];
      if (files.length > 0) {
        const formData = new FormData();
        formData.append("conversation", payload.conversation);
        formData.append("content", payload.content);
        formData.append("message_type", files.some((file) => file.type.startsWith("image/")) ? "image" : "file");
        files.forEach((file) => formData.append("attachments", file, file.name));
        return apiFetch("/chat/messages/", {
          method: "POST",
          body: formData,
        });
      }
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
    if (!open) {
      autoGreetingSentRef.current = false;
      return;
    }
    if (!hasToken) return;
    if (activeConversation.isLoading) return;
    if (conversationId) return;
    if (createConversation.isPending) return;
    if (autoGreetingSentRef.current) return;

    autoGreetingSentRef.current = true;
    createConversation.mutate("");
  }, [
    activeConversation.isLoading,
    conversationId,
    createConversation,
    hasToken,
    open,
  ]);

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

  React.useEffect(() => {
    resizeComposer();
  }, [input, resizeComposer]);

  const messages = conversationDetail.data?.messages || activeConversation.data?.messages || [];
  const assignedAgent = conversationDetail.data?.agent || activeConversation.data?.agent;
  const receiverName = assignedAgent?.display_name || "Support team";
  const isSubmitting = sendMessage.isPending || createConversation.isPending;

  const handlePickFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;
    setPendingFiles((previous) => [...previous, ...selectedFiles].slice(0, 5));
    event.target.value = "";
  };

  const removePendingFile = (index: number) => {
    setPendingFiles((previous) => previous.filter((_, fileIndex) => fileIndex !== index));
  };

  const handleSend = async () => {
    const text = input.trim();
    const files = pendingFiles;
    if ((!text && files.length === 0) || !hasToken) return;

    const draftedText = input;
    const draftedFiles = pendingFiles;
    setInput("");
    setPendingFiles([]);

    try {
      let targetConversationId = conversationId;
      const startedWithoutConversation = !targetConversationId;

      if (!targetConversationId) {
        autoGreetingSentRef.current = true;
        const createdConversation = await createConversation.mutateAsync(text);
        targetConversationId = createdConversation.id;

        if (files.length === 0) return;
      }

      if (!targetConversationId) return;

      const fallbackAttachmentLabel =
        files.length === 1
          ? `[Attachment: ${files[0].name}]`
          : `[Attachments: ${files.length} files]`;
      const contentForSend =
        files.length > 0
          ? text && !startedWithoutConversation
            ? text
            : fallbackAttachmentLabel
          : text;

      await sendMessage.mutateAsync({
        conversation: targetConversationId,
        content: contentForSend,
        files,
      });
    } catch (error) {
      setInput(draftedText);
      setPendingFiles(draftedFiles);
      throw error;
    }
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
        <>
          <button
            type="button"
            className="fixed inset-0 z-40 bg-black/35 backdrop-blur-[2px]"
            onClick={() => setOpen(false)}
            aria-label="Close support chat"
          />
          <div className="chat-widget-mobile-open-offset fixed inset-x-2 bottom-2 z-50 sm:inset-x-auto sm:bottom-6 sm:right-6">
            <div className="chat-widget-mobile-panel flex h-[min(90dvh,43rem)] min-h-[24rem] w-full flex-col overflow-hidden rounded-2xl border border-border bg-card p-3 shadow-xl sm:h-[44rem] sm:w-96 sm:p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <div className="relative h-7 w-7 overflow-hidden rounded-full bg-muted">
                  {assignedAgent?.avatar_url ? (
                    <Image
                      src={assignedAgent.avatar_url}
                      alt={receiverName}
                      fill
                      sizes="28px"
                      unoptimized
                      loading="lazy"
                      decoding="async"
                      className="object-cover"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-[11px] font-semibold text-foreground/60">
                      {initials(receiverName)}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-sm font-semibold leading-tight">{receiverName}</p>
                  <p className="text-[11px] text-foreground/60">
                    Support chat{" "}
                    {wsState === "open" ? " - Live" : wsState === "connecting" ? " - Connecting..." : " - Offline"}
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-background/80 text-foreground/70 shadow-sm transition hover:bg-muted hover:text-foreground"
                onClick={() => setOpen(false)}
                aria-label="Close chat"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {!hasToken ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border/70 bg-background/60 p-4 text-center">
                <p className="text-sm text-foreground/70">
                  Sign in to start a secure support chat with your account.
                </p>
                <Link
                  href="/account/login/"
                  className="inline-flex items-center justify-center rounded-full bg-primary px-4 py-2 text-xs font-semibold text-white"
                >
                  Sign in
                </Link>
              </div>
            ) : (
              <>
                <div
                  className="scrollbar-thin min-h-0 flex-1 space-y-3 overflow-y-auto -mr-2 pr-3 text-sm"
                  style={{ scrollbarGutter: "stable" }}
                >
                  {messages.length === 0 ? (
                    <p className="text-foreground/60">Start a conversation.</p>
                  ) : (
                    messages.map((msg) => {
                      const isMine = msg.is_from_customer;
                      const avatarUrl = msg.sender_avatar_url || msg.sender?.avatar_url || assignedAgent?.avatar_url || null;
                      return (
                        <div
                          key={msg.id}
                          className={cn("flex gap-2", isMine ? "justify-end" : "justify-start")}
                        >
                          {!isMine ? (
                            <div className="relative h-7 w-7 shrink-0 overflow-hidden rounded-full bg-muted">
                              {avatarUrl ? (
                                <Image
                                  src={avatarUrl}
                                  alt={receiverName}
                                  fill
                                  sizes="28px"
                                  unoptimized
                                  loading="lazy"
                                  decoding="async"
                                  className="object-cover"
                                />
                              ) : (
                                <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-foreground/60">
                                  {initials(receiverName)}
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
                            <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                            {msg.attachments?.length ? (
                              <div className="mt-2 space-y-1.5">
                                {msg.attachments.map((attachment) => {
                                  const fileUrl = attachment.download_url || attachment.file;
                                  if (!fileUrl) return null;
                                  return (
                                    <a
                                      key={attachment.id}
                                      href={fileUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className={cn(
                                        "flex items-center justify-between gap-2 rounded-md border px-2 py-1 text-[11px]",
                                        isMine
                                          ? "border-white/25 bg-white/10 text-white hover:bg-white/15"
                                          : "border-border bg-background/70 text-foreground hover:bg-background"
                                      )}
                                    >
                                      <span className="flex min-w-0 items-center gap-1.5">
                                        <Paperclip className="h-3 w-3 shrink-0" />
                                        <span className="truncate">{attachment.file_name || "Attachment"}</span>
                                      </span>
                                      <span className={cn("shrink-0", isMine ? "text-white/80" : "text-foreground/60")}>
                                        {formatFileSize(attachment.file_size)}
                                      </span>
                                    </a>
                                  );
                                })}
                              </div>
                            ) : null}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
                <div className="mt-3 space-y-2">
                  {pendingFiles.length > 0 ? (
                    <div className="scrollbar-thin max-h-20 space-y-1 overflow-y-auto pr-1">
                      {pendingFiles.map((file, index) => (
                        <div
                          key={`${file.name}-${file.size}-${index}`}
                          className="flex items-center justify-between rounded-md border border-border bg-background/70 px-2 py-1 text-xs"
                        >
                          <span className="truncate text-foreground/80">
                            {file.name} {formatFileSize(file.size) ? `(${formatFileSize(file.size)})` : ""}
                          </span>
                          <button
                            type="button"
                            className="ml-2 flex h-5 w-5 min-h-5 min-w-5 items-center justify-center rounded-full text-foreground/70 hover:bg-muted hover:text-foreground"
                            onClick={() => removePendingFile(index)}
                            aria-label={`Remove ${file.name}`}
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={handlePickFiles}
                    disabled={isSubmitting}
                  />
                  <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="flex h-10 w-10 min-h-10 min-w-10 items-center justify-center rounded-lg border border-border bg-background text-foreground/70 transition hover:bg-muted hover:text-foreground disabled:opacity-60"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isSubmitting}
                    aria-label="Add files"
                  >
                    <Paperclip className="h-4 w-4" />
                  </button>
                  <textarea
                    ref={composerRef}
                    rows={1}
                    className="scrollbar-thin h-10 min-h-10 flex-1 resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm leading-5"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        handleSend();
                      }
                    }}
                    placeholder="Type a message"
                    disabled={isSubmitting}
                  />
                  <button
                    type="button"
                    className="flex h-10 min-h-10 w-10 min-w-10 items-center justify-center rounded-lg bg-primary text-white disabled:opacity-60"
                    onClick={handleSend}
                    disabled={isSubmitting}
                    aria-label="Send message"
                  >
                    <SendHorizontal className="h-4 w-4" />
                  </button>
                </div>
                </div>
              </>
            )}
            </div>
          </div>
        </>
      ) : null}
    </>
  );
}
