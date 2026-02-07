"use client";

import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";

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
  const base = (process.env.NEXT_PUBLIC_WS_BASE_URL || "").replace(/\/$/, "");
  if (!base) return null;
  const normalizedPath = base.endsWith("/ws") ? path.replace(/^\/ws/, "") : path;
  return `${base}${normalizedPath}`;
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const socketsRef = React.useRef<Record<string, WebSocket | null>>({});
  const reconnectTimers = React.useRef<Record<string, ReturnType<typeof setTimeout> | null>>({});
  const connectRef = React.useRef<(channel: string) => void>(() => {});
  const [status, setStatus] = React.useState<Record<string, "connecting" | "open" | "closed" | "error">>({});
  const [lastMessage, setLastMessage] = React.useState<Record<string, unknown>>({});

  const connect = React.useCallback(
    (channel: string) => {
      const path = CHANNELS[channel];
      if (!path) return;
      const url = buildWsUrl(path);
      if (!url) return;

      setStatus((prev) => ({ ...prev, [channel]: "connecting" }));

      const ws = new WebSocket(url);
      socketsRef.current[channel] = ws;

      ws.onopen = () => {
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
        setStatus((prev) => ({ ...prev, [channel]: "closed" }));
        if (reconnectTimers.current[channel]) {
          clearTimeout(reconnectTimers.current[channel] as ReturnType<typeof setTimeout>);
        }
        reconnectTimers.current[channel] = setTimeout(
          () => connectRef.current(channel),
          3000
        );
      };
    },
    [queryClient]
  );

  React.useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  React.useEffect(() => {
    const sockets = socketsRef.current;
    const timers = reconnectTimers.current;
    Object.keys(CHANNELS).forEach((channel) => connect(channel));
    return () => {
      Object.values(sockets).forEach((socket) => socket?.close());
      Object.values(timers).forEach((timer) => timer && clearTimeout(timer));
    };
  }, [connect]);

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
