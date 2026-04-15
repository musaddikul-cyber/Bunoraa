"use client";

import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl as buildGlobalWsUrl } from "@/lib/ws";
import { useAuthContext } from "@/components/providers/AuthProvider";

type WebSocketContextValue = {
  send: (channel: string, payload: unknown) => void;
  lastMessage: Record<string, unknown>;
  status: Record<string, "connecting" | "open" | "closed" | "error">;
};

const WebSocketContext = React.createContext<WebSocketContextValue | undefined>(
  undefined
);

const CHANNELS: Record<string, string> = {
  notifications: "/ws/notifications/",
  cart: "/ws/cart/",
  search: "/ws/search/",
  analytics: "/ws/analytics/",
};

function buildWsUrl(path: string) {
  const enabled = (process.env.NEXT_PUBLIC_WS_ENABLED || "").toLowerCase() === "true";
  if (!enabled) return null;
  return buildGlobalWsUrl(path, getAccessToken());
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const { hasToken, profileQuery } = useAuthContext();
  const queryClient = useQueryClient();
  const socketsRef = React.useRef<Record<string, WebSocket | null>>({});
  const reconnectTimers = React.useRef<Record<string, ReturnType<typeof setTimeout> | null>>({});
  const reconnectAttempts = React.useRef<Record<string, number>>({});
  const activeChannelsRef = React.useRef<string[]>([]);
  const isMountedRef = React.useRef(true);
  const connectRef = React.useRef<(channel: string) => void>(() => {});
  const [status, setStatus] = React.useState<Record<string, "connecting" | "open" | "closed" | "error">>({});
  const [lastMessage, setLastMessage] = React.useState<Record<string, unknown>>({});
  const isAdmin = Boolean(profileQuery.data?.is_staff || profileQuery.data?.is_superuser);

  const activeChannels = React.useMemo(() => {
    if (!hasToken) return [];
    return Object.keys(CHANNELS).filter((channel) => channel !== "analytics" || isAdmin);
  }, [hasToken, isAdmin]);

  const connect = React.useCallback(
    (channel: string) => {
      if (typeof navigator !== "undefined" && !navigator.onLine) return;
      const path = CHANNELS[channel];
      if (!path) return;
      const existingSocket = socketsRef.current[channel];
      if (
        existingSocket &&
        (existingSocket.readyState === WebSocket.CONNECTING ||
          existingSocket.readyState === WebSocket.OPEN)
      ) {
        return;
      }
      const url = buildWsUrl(path);
      if (!url) return;

      setStatus((prev) => ({ ...prev, [channel]: "connecting" }));

      const ws = new WebSocket(url);
      socketsRef.current[channel] = ws;

      ws.onopen = () => {
        reconnectAttempts.current[channel] = 0;
        setStatus((prev) => ({ ...prev, [channel]: "open" }));
      };

      ws.onmessage = (event) => {
        let payload: unknown = event.data;
        try {
          payload = JSON.parse(event.data);
        } catch {
          payload = event.data;
        }
        setLastMessage((prev) => ({ ...prev, [channel]: payload }));

        if (channel === "cart") {
          queryClient.invalidateQueries({ queryKey: ["cart"] });
          queryClient.invalidateQueries({ queryKey: ["cart", "summary"] });
        }
        if (channel === "notifications") {
          queryClient.invalidateQueries({ queryKey: ["notifications"] });
          queryClient.invalidateQueries({ queryKey: ["notifications", "unread"] });
        }
        if (channel === "search") {
          if (payload && typeof payload === "object" && "suggestions" in (payload as object)) {
            queryClient.setQueryData(["search", "suggestions"], payload);
          } else {
            queryClient.invalidateQueries({ queryKey: ["search", "suggestions"] });
          }
        }
        if (channel === "analytics") {
          queryClient.invalidateQueries({ queryKey: ["analytics"] });
        }
      };

      ws.onerror = () => {
        setStatus((prev) => ({ ...prev, [channel]: "error" }));
      };

      ws.onclose = () => {
        if (socketsRef.current[channel] === ws) {
          socketsRef.current[channel] = null;
        }
        setStatus((prev) => ({ ...prev, [channel]: "closed" }));
        if (!isMountedRef.current) return;
        if (!activeChannelsRef.current.includes(channel)) return;
        if (reconnectTimers.current[channel]) {
          clearTimeout(reconnectTimers.current[channel] as ReturnType<typeof setTimeout>);
        }
        const attempt = (reconnectAttempts.current[channel] || 0) + 1;
        reconnectAttempts.current[channel] = attempt;
        const baseDelay = Math.min(30000, 1000 * 2 ** Math.min(5, attempt));
        const jitter = Math.floor(Math.random() * 400);
        const nextDelay = baseDelay + jitter;
        reconnectTimers.current[channel] = setTimeout(
          () => connectRef.current(channel),
          nextDelay
        );
      };
    },
    [queryClient]
  );

  React.useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  React.useEffect(() => {
    activeChannelsRef.current = activeChannels;
    activeChannels.forEach((channel) => connect(channel));
    Object.keys(CHANNELS).forEach((channel) => {
      if (activeChannels.includes(channel)) return;
      if (reconnectTimers.current[channel]) {
        clearTimeout(reconnectTimers.current[channel] as ReturnType<typeof setTimeout>);
        reconnectTimers.current[channel] = null;
      }
      const socket = socketsRef.current[channel];
      if (!socket) return;
      socket.onclose = null;
      socket.onerror = null;
      socket.onmessage = null;
      socket.close(1000, "Channel disabled");
      socketsRef.current[channel] = null;
      setStatus((prev) => ({ ...prev, [channel]: "closed" }));
    });
  }, [activeChannels, connect]);

  React.useEffect(() => {
    const handleOnline = () => {
      activeChannelsRef.current.forEach((channel) => connectRef.current(channel));
    };
    window.addEventListener("online", handleOnline);
    return () => {
      window.removeEventListener("online", handleOnline);
    };
  }, []);

  React.useEffect(() => {
    const reconnectTimersCurrent = reconnectTimers.current;
    const socketsCurrent = socketsRef.current;
    return () => {
      isMountedRef.current = false;
      Object.values(reconnectTimersCurrent).forEach((timer) => timer && clearTimeout(timer));
      Object.values(socketsCurrent).forEach((socket) => socket?.close(1000, "Provider unmounted"));
    };
  }, []);

  const send = React.useCallback((channel: string, payload: unknown) => {
    const socket = socketsRef.current[channel];
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const message = typeof payload === "string" ? payload : JSON.stringify(payload);
    socket.send(message);
  }, []);

  return (
    <WebSocketContext.Provider value={{ send, lastMessage, status }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const ctx = React.useContext(WebSocketContext);
  if (!ctx) {
    throw new Error("useWebSocket must be used within WebSocketProvider");
  }
  return ctx;
}
