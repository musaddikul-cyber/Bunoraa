"use client";

import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

const statusOptions = [
  { value: "all", label: "All" },
  { value: "open", label: "Open" },
  { value: "waiting", label: "Waiting" },
  { value: "active", label: "Active" },
  { value: "resolved", label: "Resolved" },
  { value: "closed", label: "Closed" },
];

type Conversation = {
  id: string;
  subject?: string | null;
  status: string;
  category?: string;
  customer_name?: string | null;
  customer_email?: string | null;
  last_message?: { content: string; created_at: string } | null;
  internal_notes?: string | null;
  messages?: Message[];
};

type Message = {
  id: string;
  content: string;
  is_from_customer: boolean;
  is_read?: boolean;
  message_type?: string;
  reactions?: Record<string, string[]>;
  created_at: string;
};

type AgentProfile = {
  id: string;
  is_online: boolean;
  is_accepting_chats: boolean;
  display_name?: string;
};

type CannedResponse = {
  id: string;
  title: string;
  content: string;
  shortcut: string;
  category?: string;
  tags?: string[];
  is_global?: boolean;
};

function buildChatWsUrl(conversationId: string) {
  const base = (process.env.NEXT_PUBLIC_WS_BASE_URL || "").replace(/\/$/, "");
  if (!base) return null;
  const path = `/ws/chat/${conversationId}/`;
  const normalizedPath = base.endsWith("/ws") ? path.replace(/^\/ws/, "") : path;
  return `${base}${normalizedPath}`;
}

export default function AgentChatPage() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = React.useState("active");
  const [search, setSearch] = React.useState("");
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [messageInput, setMessageInput] = React.useState("");
  const [noteDraft, setNoteDraft] = React.useState("");
  const [typingUsers, setTypingUsers] = React.useState<Record<string, boolean>>({});
  const [emailSubject, setEmailSubject] = React.useState("");
  const [emailBody, setEmailBody] = React.useState("");
  const [cannedQuery, setCannedQuery] = React.useState("");
  const socketRef = React.useRef<WebSocket | null>(null);
  const typingTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  const agentProfile = useQuery({
    queryKey: ["agent", "me"],
    queryFn: async () => {
      const response = await apiFetch<AgentProfile>("/chat/agents/me/");
      return response.data;
    },
  });

  const queue = useQuery({
    queryKey: ["agent", "queue"],
    queryFn: async () => {
      const response = await apiFetch<Conversation[]>("/chat/conversations/queue/");
      return response.data;
    },
  });

  const conversations = useQuery({
    queryKey: ["agent", "conversations", statusFilter],
    queryFn: async () => {
      const params = statusFilter !== "all" ? { status: statusFilter } : undefined;
      const response = await apiFetch<Conversation[]>("/chat/conversations/", { params });
      return response.data;
    },
  });

  const cannedResponses = useQuery({
    queryKey: ["agent", "canned"],
    queryFn: async () => {
      const response = await apiFetch<CannedResponse[]>("/chat/canned-responses/");
      return response.data;
    },
  });

  const selectedConversation = useQuery({
    queryKey: ["agent", "conversation", selectedId],
    queryFn: async () => {
      if (!selectedId) return null;
      const response = await apiFetch<Conversation>(`/chat/conversations/${selectedId}/`);
      return response.data;
    },
    enabled: Boolean(selectedId),
  });

  React.useEffect(() => {
    const notes = selectedConversation.data?.internal_notes || "";
    setNoteDraft(notes);
  }, [selectedConversation.data?.internal_notes]);

  React.useEffect(() => {
    const conversation = selectedConversation.data;
    if (!conversation?.messages) return;
    const unread = conversation.messages.filter(
      (msg) => msg.is_from_customer && !msg.is_read
    );
    if (!unread.length) return;
    unread.slice(0, 5).forEach((msg) => {
      apiFetch(`/chat/messages/${msg.id}/mark_read/`, { method: "POST" }).catch(() => null);
    });
  }, [selectedConversation.data]);

  React.useEffect(() => {
    if (!selectedId) return;
    setTypingUsers({});
    const url = buildChatWsUrl(selectedId);
    if (!url) return;
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "typing") {
          setTypingUsers((prev) => ({
            ...prev,
            [payload.user_id]: payload.is_typing,
          }));
        }
      } catch {
        // ignore
      }
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations", statusFilter] });
      queryClient.invalidateQueries({ queryKey: ["agent", "queue"] });
    };

    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [selectedId, queryClient, statusFilter]);

  const handleTyping = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "typing_start" }));
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ type: "typing_stop" }));
      }
    }, 1500);
  };

  const updateStatus = useMutation({
    mutationFn: async (payload: { is_online?: boolean; is_accepting_chats?: boolean }) => {
      return apiFetch<AgentProfile>("/chat/agents/me/status/", {
        method: "POST",
        body: payload,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", "me"] });
    },
  });

  const assignConversation = useMutation({
    mutationFn: async (conversationId: string) => {
      return apiFetch<Conversation>(`/chat/conversations/${conversationId}/assign/`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", "queue"] });
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations", statusFilter] });
    },
  });

  const sendMessage = useMutation({
    mutationFn: async (payload: { conversation: string; content: string }) => {
      return apiFetch<Message>("/chat/messages/", {
        method: "POST",
        body: { conversation: payload.conversation, content: payload.content, message_type: "text" },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
      queryClient.invalidateQueries({ queryKey: ["agent", "conversations", statusFilter] });
    },
  });

  const saveNotes = useMutation({
    mutationFn: async (payload: { conversationId: string; internal_notes: string }) => {
      return apiFetch<Conversation>(`/chat/conversations/${payload.conversationId}/internal-notes/`, {
        method: "POST",
        body: { internal_notes: payload.internal_notes },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
    },
  });

  const sendEmail = useMutation({
    mutationFn: async (payload: { conversationId: string; subject: string; body: string }) => {
      return apiFetch<Conversation>(`/chat/conversations/${payload.conversationId}/email/`, {
        method: "POST",
        body: { subject: payload.subject, text_body: payload.body },
      });
    },
    onSuccess: () => {
      setEmailBody("");
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
    },
  });

  const reactToMessage = useMutation({
    mutationFn: async (payload: { messageId: string; emoji: string }) => {
      return apiFetch<Message>(`/chat/messages/${payload.messageId}/react/`, {
        method: "POST",
        body: { emoji: payload.emoji },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent", "conversation", selectedId] });
    },
  });

  const filteredConversations = (conversations.data || []).filter((conversation) => {
    const query = search.trim().toLowerCase();
    if (!query) return true;
    const haystack = [
      conversation.subject,
      conversation.customer_name,
      conversation.customer_email,
      conversation.last_message?.content,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });

  const filteredCanned = (cannedResponses.data || []).filter((item) => {
    const query = cannedQuery.trim().toLowerCase();
    if (!query) return true;
    return (
      item.title.toLowerCase().includes(query) ||
      item.shortcut.toLowerCase().includes(query) ||
      item.content.toLowerCase().includes(query)
    );
  });

  const currentMessages = selectedConversation.data?.messages || [];
  const typingList = Object.entries(typingUsers)
    .filter(([, value]) => value)
    .map(([id]) => id);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-7xl px-6 py-12">
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
              Agent Console
            </p>
            <h1 className="text-3xl font-semibold">Chat Operations</h1>
            <p className="text-sm text-foreground/70">
              Manage live conversations and email support in one place.
            </p>
          </div>
          <Card variant="bordered" className="flex flex-wrap items-center gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Status</p>
              <p className="text-sm font-semibold">
                {agentProfile.data?.display_name || "Agent"}
              </p>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-foreground/70">
                <input
                  type="checkbox"
                  checked={agentProfile.data?.is_online ?? false}
                  onChange={(event) => updateStatus.mutate({ is_online: event.target.checked })}
                />
                Online
              </label>
              <label className="flex items-center gap-2 text-sm text-foreground/70">
                <input
                  type="checkbox"
                  checked={agentProfile.data?.is_accepting_chats ?? false}
                  onChange={(event) =>
                    updateStatus.mutate({ is_accepting_chats: event.target.checked })
                  }
                />
                Accepting
              </label>
            </div>
            {agentProfile.isLoading && (
              <p className="text-xs text-foreground/50">Loading status...</p>
            )}
            {agentProfile.isError && (
              <p className="text-xs text-foreground/50">Unable to load agent profile.</p>
            )}
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-[280px_360px_1fr]">
          <Card variant="bordered" className="space-y-6">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/60">
                Queue
              </h2>
              <div className="mt-3 space-y-3">
                {queue.isLoading && (
                  <p className="text-xs text-foreground/60">Loading queue...</p>
                )}
                {queue.isError && (
                  <p className="text-xs text-foreground/60">Unable to load queue.</p>
                )}
                {(queue.data || []).map((conversation) => (
                  <div
                    key={conversation.id}
                    role="button"
                    tabIndex={0}
                    className="w-full rounded-xl border border-border bg-card/60 p-3 text-left transition hover:border-primary/40"
                    onClick={() => setSelectedId(conversation.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedId(conversation.id);
                      }
                    }}
                  >
                    <p className="text-sm font-semibold">
                      {conversation.customer_name ||
                        conversation.customer_email ||
                        "Guest"}
                    </p>
                    <p className="text-xs text-foreground/60">
                      {conversation.subject || "No subject"}
                    </p>
                    <div className="mt-2 flex items-center justify-between text-xs text-foreground/60">
                      <span>{conversation.category}</span>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(event) => {
                          event.stopPropagation();
                          assignConversation.mutate(conversation.id);
                        }}
                      >
                        Assign
                      </Button>
                    </div>
                  </div>
                ))}
                {queue.data?.length === 0 && !queue.isLoading && (
                  <p className="text-xs text-foreground/50">No one waiting.</p>
                )}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/60">
                Canned Replies
              </h3>
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-xs"
                placeholder="Search canned..."
                value={cannedQuery}
                onChange={(event) => setCannedQuery(event.target.value)}
              />
              <div className="mt-3 max-h-64 space-y-2 overflow-y-auto">
                {cannedResponses.isLoading && (
                  <p className="text-xs text-foreground/60">Loading canned replies...</p>
                )}
                {cannedResponses.isError && (
                  <p className="text-xs text-foreground/60">Unable to load canned replies.</p>
                )}
                {filteredCanned.map((item) => (
                  <button
                    key={item.id}
                    className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-left text-xs text-foreground/80 hover:border-primary/40"
                    onClick={() => setMessageInput(item.content)}
                  >
                    <p className="font-semibold text-foreground">{item.title}</p>
                    <p className="text-[10px] text-foreground/50">{item.shortcut}</p>
                  </button>
                ))}
              </div>
            </div>
          </Card>

          <Card variant="bordered" className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground/60">
                Conversations
              </h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {statusOptions.map((option) => (
                <button
                  key={option.value}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition",
                    statusFilter === option.value
                      ? "border-primary/60 bg-primary/10 text-primary"
                      : "border-border text-foreground/60 hover:border-primary/40"
                  )}
                  onClick={() => setStatusFilter(option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-xs"
              placeholder="Search conversations"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <div className="mt-2 max-h-[60vh] space-y-3 overflow-y-auto">
              {conversations.isLoading && (
                <p className="text-xs text-foreground/60">Loading conversations...</p>
              )}
              {conversations.isError && (
                <p className="text-xs text-foreground/60">Unable to load conversations.</p>
              )}
              {filteredConversations.map((conversation) => (
                <button
                  key={conversation.id}
                  className={cn(
                    "w-full rounded-xl border p-3 text-left transition",
                    selectedId === conversation.id
                      ? "border-primary/60 bg-primary/10"
                      : "border-border bg-card/60 hover:border-primary/30"
                  )}
                  onClick={() => setSelectedId(conversation.id)}
                >
                  <div className="flex items-center justify-between text-xs text-foreground/50">
                    <span className="uppercase tracking-[0.2em]">
                      {conversation.status}
                    </span>
                    <span>{conversation.category}</span>
                  </div>
                  <p className="mt-1 text-sm font-semibold">
                    {conversation.customer_name ||
                      conversation.customer_email ||
                      "Guest"}
                  </p>
                  <p className="text-xs text-foreground/60">
                    {conversation.subject || "No subject"}
                  </p>
                  {conversation.last_message?.content && (
                    <p className="mt-2 max-h-10 overflow-hidden text-ellipsis text-xs text-foreground/60">
                      {conversation.last_message.content}
                    </p>
                  )}
                </button>
              ))}
            </div>
          </Card>

          <Card variant="bordered" className="flex h-full flex-col">
            {!selectedConversation.data && !selectedConversation.isLoading && (
              <div className="flex h-full flex-col items-center justify-center text-center text-foreground/60">
                <p className="text-sm">Select a conversation to begin.</p>
              </div>
            )}
            {selectedConversation.isLoading && (
              <div className="flex h-full flex-col items-center justify-center text-center text-foreground/60">
                <p className="text-sm">Loading conversation...</p>
              </div>
            )}
            {selectedConversation.isError && (
              <div className="flex h-full flex-col items-center justify-center text-center text-foreground/60">
                <p className="text-sm">Unable to load conversation.</p>
              </div>
            )}
            {selectedConversation.data && (
              <div className="flex h-full flex-col">
                <div className="mb-4 border-b border-border pb-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">Live</p>
                  <h2 className="text-2xl font-semibold">
                    {selectedConversation.data.subject || "Conversation"}
                  </h2>
                  <p className="text-sm text-foreground/70">
                    {selectedConversation.data.customer_name ||
                      selectedConversation.data.customer_email ||
                      "Guest"}
                  </p>
                  {typingList.length > 0 && (
                    <p className="mt-2 text-xs text-primary">
                      Customer is typing...
                    </p>
                  )}
                </div>

                <div className="flex-1 space-y-3 overflow-y-auto pr-2">
                  {currentMessages.map((msg) => (
                    <div
                      key={msg.id}
                      className={cn(
                        "rounded-2xl border p-4 text-sm",
                        msg.is_from_customer
                          ? "border-border bg-muted/50"
                          : "border-primary/20 bg-primary/10"
                      )}
                    >
                      <div className="flex items-center justify-between text-xs text-foreground/50">
                        <span>{msg.message_type || "text"}</span>
                        <span>{new Date(msg.created_at).toLocaleTimeString()}</span>
                      </div>
                      <p className="mt-2 whitespace-pre-wrap text-foreground/80">{msg.content}</p>
                      <div className="mt-2 flex items-center gap-2">
                        {["??", "?", "?"].map((emoji) => (
                          <button
                            key={emoji}
                            className="rounded-full border border-border px-2 py-1 text-xs text-foreground/70"
                            onClick={() => reactToMessage.mutate({ messageId: msg.id, emoji })}
                          >
                            {emoji}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-xl border border-border bg-card/60 p-4">
                  <textarea
                    className="h-24 w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm"
                    value={messageInput}
                    onChange={(event) => {
                      setMessageInput(event.target.value);
                      handleTyping();
                    }}
                    placeholder="Type a response"
                  />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => {
                        if (!selectedId || !messageInput.trim()) return;
                        sendMessage.mutate({ conversation: selectedId, content: messageInput.trim() });
                        setMessageInput("");
                      }}
                    >
                      Send Message
                    </Button>
                  </div>
                </div>

                <div className="mt-4 grid gap-4 lg:grid-cols-2">
                  <Card variant="bordered" className="p-4">
                    <h3 className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                      Internal Notes
                    </h3>
                    <textarea
                      className="mt-2 h-24 w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm"
                      value={noteDraft}
                      onChange={(event) => setNoteDraft(event.target.value)}
                      placeholder="Add internal notes"
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-3"
                      onClick={() => {
                        if (!selectedId) return;
                        saveNotes.mutate({ conversationId: selectedId, internal_notes: noteDraft });
                      }}
                    >
                      Save Notes
                    </Button>
                  </Card>

                  <Card variant="bordered" className="p-4">
                    <h3 className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                      Email Reply
                    </h3>
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                      value={emailSubject}
                      onChange={(event) => setEmailSubject(event.target.value)}
                      placeholder="Subject"
                    />
                    <textarea
                      className="mt-2 h-24 w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm"
                      value={emailBody}
                      onChange={(event) => setEmailBody(event.target.value)}
                      placeholder="Write email reply"
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-3"
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
                  </Card>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
