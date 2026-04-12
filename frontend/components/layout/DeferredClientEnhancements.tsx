"use client";

import * as React from "react";
import dynamic from "next/dynamic";

const PageViewTracker = dynamic(
  () => import("@/components/analytics/PageViewTracker").then((mod) => mod.PageViewTracker),
  { ssr: false }
);
const CompareTray = dynamic(
  () => import("@/components/products/CompareTray").then((mod) => mod.CompareTray),
  { ssr: false }
);
const ChatWidget = dynamic(
  () => import("@/components/chat/ChatWidget").then((mod) => mod.ChatWidget),
  { ssr: false }
);

export function DeferredClientEnhancements() {
  const [isReady, setIsReady] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let idleId: number | null = null;

    const enable = () => setIsReady(true);
    const idleApi = globalThis as typeof globalThis & {
      requestIdleCallback?: (callback: IdleRequestCallback, options?: IdleRequestOptions) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof idleApi.requestIdleCallback === "function") {
      idleId = idleApi.requestIdleCallback(enable, { timeout: 2500 });
      return () => {
        if (idleId !== null && typeof idleApi.cancelIdleCallback === "function") {
          idleApi.cancelIdleCallback(idleId);
        }
      };
    }

    timeoutId = setTimeout(enable, 1200);
    return () => {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  if (!isReady) return null;

  return (
    <>
      <PageViewTracker />
      <CompareTray />
      <ChatWidget />
    </>
  );
}
