"use client";

import * as React from "react";
import Image from "next/image";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Paperclip, X } from "lucide-react";
import { AuthGate } from "@/components/auth/AuthGate";
import { useAuthContext } from "@/components/providers/AuthProvider";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { getAccessToken } from "@/lib/auth";
import { apiFetch } from "@/lib/api";
import { buildWsUrl } from "@/lib/ws";
import { cn } from "@/lib/utils";

const statusOptions = ["all", "open", "waiting", "active", "resolved", "closed"];

type MessageAttachment = {
  id: string;
  file?: string | null;
  file_name?: string;
  file_type?: string;
  file_size?: number;
  download_url?: string | null;
};

type Message = {
  id: string;
  content: string;
  is_from_customer: boolean;
  is_read?: boolean;
  message_type?: string;
  sender_display_name?: string | null;
  sender_avatar_url?: string | null;
  attachments?: MessageAttachment[];
  created_at: string;
};

type Conversation = {
  id: string;
  subject?: string | null;
  status: string;
  category?: string | null;
  customer_name?: string | null;
  customer_email?: string | null;
  customer_avatar_url?: string | null;
  last_message?: { content: string; created_at: string } | null;
  internal_notes?: string | null;
  messages?: Message[];
};

type AgentProfile = {
  id: string;
  is_online: boolean;
  is_accepting_chats: boolean;
  display_name?: string | null;
  avatar_url?: string | null;
};

type CannedResponse = { id: string; title: string; content: string; shortcut: string };

function buildChatWsUrl(conversationId: string) {
  return buildWsUrl(`/ws/chat/${conversationId}/`, getAccessToken());
}

function initials(name?: string | null) {
  const safe = (name || "").trim();
  if (!safe) return "?";
  return safe
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() || "")
    .join("");
}

function formatFileSize(size?: number) {
  if (!size || size <= 0) return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function Avatar({ name, url }: { name?: string | null; url?: string | null }) {
  return (
    <div className="relative h-7 w-7 shrink-0 overflow-hidden rounded-full bg-muted">
      {url ? (
        <Image
          src={url}
          alt={name || "User"}
          fill
          sizes="28px"
          unoptimized
          loading="lazy"
          decoding="async"
          className="object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center text-[10px] font-semibold text-foreground/60">
          {initials(name)}
        </div>
      )}
    </div>
  );
}

function AgentChatConsole() {
  const queryClient = useQueryClient();
  const { profileQuery } = useAuthContext();
  const wsEnabled = (process.env.NEXT_PUBLIC_WS_ENABLED || "").toLowerCase() === "true";
  const profile = profileQuery.data;
  const isStaff = Boolean(profile?.is_staff || profile?.is_superuser);

  const [statusFilter, setStatusFilter] = React.useState("active");
  const [search, setSearch] = React.useState("");
  const [customerScope, setCustomerScope] = React.useState("");
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [messageInput, setMessageInput] = React.useState("");
  const [pendingFiles, setPendingFiles] = React.useState<File[]>([]);
  const [noteDraft, setNoteDraft] = React.useState("");
  const [emailSubject, setEmailSubject] = React.useState("");
  const [emailBody, setEmailBody] = React.useState("");
  const [cannedQuery, setCannedQuery] = React.useState("");
  const [targetUserId, setTargetUserId] = React.useState("");
  const [targetSubject, setTargetSubject] = React.useState("Support");
  const [typingUsers, setTypingUsers] = React.useState<Record<string, boolean>>({});
  const socketRef = React.useRef<WebSocket | null>(null);
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

  const agentProfile = useQuery({
    queryKey: ["agent", "me"],
    queryFn: async () => (await apiFetch<AgentProfile>("/chat/agents/me/")).data,
    enabled: isStaff,
    retry: false,
  });

  const queue = useQuery({
    queryKey: ["agent", "queue"],
    queryFn: async () => (await apiFetch<Conversation[]>("/chat/conversations/queue/")).data,
    enabled: isStaff,
  });

  const conversations = useQuery({
    queryKey: ["agent", "conversations", statusFilter, customerScope],
    queryFn: async () => {
      const params: Record<string, string> = {};
      if (statusFilter !== "all") params.status = statusFilter;
      if (customerScope.trim()) params.customer_id = customerScope.trim();
      return (await apiFetch<Conversation[]>("/chat/conversations/", { params })).data;
    },
    enabled: isStaff,
  });

  const canned = useQuery({
    queryKey: ["agent", "canned"],
    queryFn: async () => (await apiFetch<CannedResponse[]>("/chat/canned-responses/")).data,
    enabled: isStaff,
  });

  const selectedConversation = useQuery({
    queryKey: ["agent", "conversation", selectedId],
    queryFn: async () =>
      selectedId ? (await apiFetch<Conversation>(`/chat/conversations/${selectedId}/`)).data : null,
    enabled: isStaff && Boolean(selectedId),
  });

  React.useEffect(() => setNoteDraft(selectedConversation.data?.internal_notes || ""), [selectedConversation.data?.internal_notes]);
  React.useEffect(() => resizeComposer(), [messageInput, resizeComposer]);
  React.useEffect(() => setPendingFiles([]), [selectedId]);

  React.useEffect(() => {
    const conversation = selectedConversation.data;
    if (!conversation?.messages?.length) return;
    const unread = conversation.messages.filter((msg) => msg.is_from_customer && !msg.is_read);
    unread.slice(0, 10).forEach((msg) => {
      apiFetch(`/chat/messages/${msg.id}/mark_read/`, {
        method: "POST",
        suppressError: true,
      }).catch(() => null);
    });
  }, [selectedConversation.data]);

  React.useEffect(() => {
    if (!wsEnabled || !selectedId || !isStaff) return;
    const url = buildChatWsUrl(selectedId);
    if (!url) return;
    const socket = new WebSocket(url);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "typing") setTypingUsers((prev) => ({ ...prev, [String(payload.user_id)]: Boolean(payload.is_typing) }));
      } catch {}
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations"] });
      queryClient.invalidateQueries({ queryKey: ["agent", "queue"] });
    };
    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [isStaff, queryClient, selectedId, wsEnabled]);

  const markTyping = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "typing_start" }));
  };

  const updateStatus = useMutation({
    mutationFn: async (payload: { is_online?: boolean; is_accepting_chats?: boolean }) =>
      apiFetch<AgentProfile>("/chat/agents/me/status/", { method: "POST", body: payload }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agent", "me"] }),
  });

  const assignConversation = useMutation({
    mutationFn: async (conversationId: string) =>
      apiFetch<Conversation>(`/chat/conversations/${conversationId}/assign/`, { method: "POST" }),
    onSuccess: (_r, conversationId) => {
      setSelectedId(conversationId);
      queryClient.invalidateQueries({ queryKey: ["agent", "queue"] });
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations"] });
    },
  });

  const startForUser = useMutation({
    mutationFn: async (payload: { user_id: string; subject: string }) =>
      (await apiFetch<Conversation>("/chat/conversations/for-user/", { method: "POST", body: payload })).data,
    onSuccess: (conversation) => {
      setSelectedId(conversation.id);
      setCustomerScope(targetUserId.trim());
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations"] });
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
        return apiFetch<Message>("/chat/messages/", { method: "POST", body: formData });
      }
      return apiFetch<Message>("/chat/messages/", { method: "POST", body: { ...payload, message_type: "text" } });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] }),
  });

  const saveNotes = useMutation({
    mutationFn: async (payload: { conversationId: string; internal_notes: string }) =>
      apiFetch<Conversation>(`/chat/conversations/${payload.conversationId}/internal-notes/`, {
        method: "POST",
        body: { internal_notes: payload.internal_notes },
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] }),
  });

  const sendEmail = useMutation({
    mutationFn: async (payload: { conversationId: string; subject: string; body: string }) =>
      apiFetch<Conversation>(`/chat/conversations/${payload.conversationId}/email/`, {
        method: "POST",
        body: { subject: payload.subject, text_body: payload.body },
      }),
    onSuccess: () => {
      setEmailBody("");
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
    },
  });

  const filteredConversations = (conversations.data || []).filter((conversation) => {
    const haystack = [conversation.subject, conversation.customer_name, conversation.customer_email, conversation.last_message?.content]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(search.toLowerCase());
  });

  const filteredCanned = (canned.data || []).filter((item) => {
    const q = cannedQuery.toLowerCase();
    return !q || item.title.toLowerCase().includes(q) || item.shortcut.toLowerCase().includes(q) || item.content.toLowerCase().includes(q);
  });

  const isSending = sendMessage.isPending;

  const handlePickFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;
    setPendingFiles((previous) => [...previous, ...selectedFiles].slice(0, 5));
    event.target.value = "";
  };

  const removePendingFile = (index: number) => {
    setPendingFiles((previous) => previous.filter((_, fileIndex) => fileIndex !== index));
  };

  const handleSendMessage = async () => {
    const text = messageInput.trim();
    if (!selectedId || (!text && pendingFiles.length === 0)) return;

    const draftedText = messageInput;
    const draftedFiles = pendingFiles;
    const fallbackAttachmentLabel =
      pendingFiles.length === 1
        ? `[Attachment: ${pendingFiles[0].name}]`
        : `[Attachments: ${pendingFiles.length} files]`;
    const content = text || fallbackAttachmentLabel;

    setMessageInput("");
    setPendingFiles([]);
    try {
      await sendMessage.mutateAsync({ conversation: selectedId, content, files: pendingFiles });
    } catch (error) {
      setMessageInput(draftedText);
      setPendingFiles(draftedFiles);
      throw error;
    }
  };

  if (profileQuery.isLoading) return <div className="p-8 text-sm text-foreground/70">Loading...</div>;
  if (!isStaff) return <div className="p-8 text-sm text-foreground/70">Staff access required.</div>;

  const messages = selectedConversation.data?.messages || [];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 lg:grid-cols-[300px_340px_1fr]">
        <Card variant="bordered" className="space-y-4">
          <div className="flex items-center gap-3">
            <Avatar name={agentProfile.data?.display_name || profile?.full_name} url={agentProfile.data?.avatar_url || profile?.avatar || null} />
            <div className="text-sm font-semibold">{agentProfile.data?.display_name || "Agent"}</div>
          </div>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={agentProfile.data?.is_online ?? false} onChange={(e) => updateStatus.mutate({ is_online: e.target.checked })} />Online</label>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={agentProfile.data?.is_accepting_chats ?? false} onChange={(e) => updateStatus.mutate({ is_accepting_chats: e.target.checked })} />Accepting chats</label>
          <input className="w-full rounded border border-border bg-card px-3 py-2 text-xs" placeholder="User UUID" value={targetUserId} onChange={(e) => setTargetUserId(e.target.value)} />
          <input className="w-full rounded border border-border bg-card px-3 py-2 text-xs" placeholder="Subject" value={targetSubject} onChange={(e) => setTargetSubject(e.target.value)} />
          <Button size="sm" onClick={() => targetUserId.trim() && startForUser.mutate({ user_id: targetUserId.trim(), subject: targetSubject || "Support" })}>Start User Chat</Button>
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-foreground/60">Queue</div>
            {(queue.data || []).map((conversation) => (
              <div key={conversation.id} className="rounded border border-border p-2 text-xs">
                <div className="truncate font-semibold">{conversation.customer_name || conversation.customer_email || "Guest"}</div>
                <Button size="sm" variant="secondary" className="mt-2" onClick={() => assignConversation.mutate(conversation.id)}>Assign</Button>
              </div>
            ))}
          </div>
          <input className="w-full rounded border border-border bg-card px-3 py-2 text-xs" placeholder="Search canned" value={cannedQuery} onChange={(e) => setCannedQuery(e.target.value)} />
          <div className="max-h-52 space-y-2 overflow-y-auto">
            {filteredCanned.map((item) => (
              <button key={item.id} className="w-full rounded border border-border px-2 py-1 text-left text-xs" onClick={() => setMessageInput(item.content)}>
                <div className="font-semibold">{item.title}</div>
                <div className="text-foreground/60">{item.shortcut}</div>
              </button>
            ))}
          </div>
        </Card>

        <Card variant="bordered" className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {statusOptions.map((status) => (
              <button key={status} className={cn("rounded-full border px-3 py-1 text-xs", statusFilter === status ? "border-primary/60 bg-primary/10 text-primary" : "border-border text-foreground/60")} onClick={() => setStatusFilter(status)}>
                {status}
              </button>
            ))}
          </div>
          <input className="w-full rounded border border-border bg-card px-3 py-2 text-xs" placeholder="Search conversations" value={search} onChange={(e) => setSearch(e.target.value)} />
          <input className="w-full rounded border border-border bg-card px-3 py-2 text-xs" placeholder="Filter customer UUID" value={customerScope} onChange={(e) => setCustomerScope(e.target.value)} />
          <div className="max-h-[70vh] space-y-2 overflow-y-auto">
            {filteredConversations.map((conversation) => (
              <button key={conversation.id} className={cn("w-full rounded border p-2 text-left", selectedId === conversation.id ? "border-primary/50 bg-primary/5" : "border-border")} onClick={() => setSelectedId(conversation.id)}>
                <div className="flex items-center gap-2">
                  <Avatar name={conversation.customer_name || conversation.customer_email} url={conversation.customer_avatar_url || null} />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold">{conversation.customer_name || conversation.customer_email || "Guest"}</div>
                    <div className="truncate text-xs text-foreground/60">{conversation.subject || "No subject"}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </Card>

        <Card variant="bordered" className="flex h-full flex-col">
          {!selectedConversation.data && <div className="p-6 text-sm text-foreground/60">Select a conversation.</div>}
          {selectedConversation.data && (
            <div className="flex h-full flex-col gap-3">
              <div className="border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <Avatar
                    name={selectedConversation.data.customer_name || selectedConversation.data.customer_email}
                    url={selectedConversation.data.customer_avatar_url || null}
                  />
                  <div>
                    <div className="text-sm font-semibold">
                      {selectedConversation.data.customer_name || selectedConversation.data.customer_email || "Guest"}
                    </div>
                    <div className="text-xs text-foreground/60">
                      {selectedConversation.data.subject || "Conversation"}
                    </div>
                  </div>
                </div>
                {Object.values(typingUsers).some(Boolean) && <div className="text-xs text-primary">Customer typing...</div>}
              </div>
              <div
                className="scrollbar-thin flex-1 space-y-3 overflow-y-auto -mr-2 pr-3"
                style={{ scrollbarGutter: "stable" }}
              >
                {messages.map((msg) => (
                  <div key={msg.id} className={cn("flex gap-2", msg.is_from_customer ? "justify-start" : "justify-end")}>
                    {msg.is_from_customer ? (
                      <Avatar
                        name={selectedConversation.data?.customer_name || selectedConversation.data?.customer_email || "Customer"}
                        url={msg.sender_avatar_url || selectedConversation.data?.customer_avatar_url || null}
                      />
                    ) : null}
                    <div
                      className={cn(
                        "max-w-[80%] rounded-2xl px-3 py-2",
                        msg.is_from_customer ? "bg-muted text-foreground" : "bg-primary text-white"
                      )}
                    >
                      <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
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
                                  msg.is_from_customer
                                    ? "border-border bg-background/70 text-foreground hover:bg-background"
                                    : "border-white/25 bg-white/10 text-white hover:bg-white/15"
                                )}
                              >
                                <span className="flex min-w-0 items-center gap-1.5">
                                  <Paperclip className="h-3 w-3 shrink-0" />
                                  <span className="truncate">{attachment.file_name || "Attachment"}</span>
                                </span>
                                <span className={cn("shrink-0", msg.is_from_customer ? "text-foreground/60" : "text-white/80")}>
                                  {formatFileSize(attachment.file_size)}
                                </span>
                              </a>
                            );
                          })}
                        </div>
                      ) : null}
                      <div className={cn("mt-1 text-[10px]", msg.is_from_customer ? "text-foreground/50" : "text-white/80")}>
                        {new Date(msg.created_at).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="space-y-2">
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
                  disabled={isSending}
                />
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="flex h-10 w-10 min-h-10 min-w-10 items-center justify-center rounded-lg border border-border bg-background text-foreground/70 transition hover:bg-muted hover:text-foreground disabled:opacity-60"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isSending}
                    aria-label="Add files"
                  >
                    <Paperclip className="h-4 w-4" />
                  </button>
                  <textarea
                    ref={composerRef}
                    rows={1}
                    className="scrollbar-thin h-10 min-h-10 flex-1 resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm leading-5"
                    value={messageInput}
                    onChange={(event) => {
                      setMessageInput(event.target.value);
                      markTyping();
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void handleSendMessage();
                      }
                    }}
                    placeholder="Type a response"
                    disabled={isSending}
                  />
                  <button
                    type="button"
                    className="h-10 min-h-10 rounded-lg bg-primary px-3 text-sm font-semibold text-white disabled:opacity-60"
                    onClick={() => void handleSendMessage()}
                    disabled={isSending}
                  >
                    Send
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="secondary" onClick={() => selectedId && saveNotes.mutate({ conversationId: selectedId, internal_notes: noteDraft })}>Save Notes</Button>
              </div>
              <textarea className="h-20 w-full resize-none rounded border border-border bg-background px-3 py-2 text-xs" value={noteDraft} onChange={(e) => setNoteDraft(e.target.value)} placeholder="Internal notes" />
              <input className="w-full rounded border border-border bg-background px-3 py-2 text-xs" value={emailSubject} onChange={(e) => setEmailSubject(e.target.value)} placeholder="Email subject" />
              <textarea className="h-20 w-full resize-none rounded border border-border bg-background px-3 py-2 text-xs" value={emailBody} onChange={(e) => setEmailBody(e.target.value)} placeholder="Email reply body" />
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  if (!selectedId || !emailBody.trim()) return;
                  sendEmail.mutate({
                    conversationId: selectedId,
                    subject: emailSubject || selectedConversation.data?.subject || "Support",
                    body: emailBody.trim(),
                  });
                }}
              >
                Send Email
              </Button>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

export default function AgentChatPage() {
  return (
    <AuthGate title="Authentication required" description="Sign in with a staff account to access chat operations." nextHref="/agent/chat/">
      <AgentChatConsole />
    </AuthGate>
  );
}
