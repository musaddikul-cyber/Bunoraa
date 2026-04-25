"use client";

import * as React from "react";
import { useEffect } from "react";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ThemeProvider } from "@/components/theme/ThemeProvider";
import { AuthProvider } from "@/components/providers/AuthProvider";
import { LocaleProvider } from "@/components/providers/LocaleProvider";
import { WebSocketProvider } from "@/components/providers/WebSocketProvider";
import { AccessibilityProvider } from "@/components/providers/AccessibilityProvider";
import { ToastProvider } from "@/components/ui/ToastProvider";
import { initPerformanceMonitoring } from "@/lib/performance";

// Initialize performance monitoring
function PerformanceMonitoring() {
  useEffect(() => {
    initPerformanceMonitoring();
  }, []);
  return null;
}

// Register service worker for offline support
function ServiceWorkerRegistration() {
  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      "serviceWorker" in navigator &&
      process.env.NODE_ENV === "production"
    ) {
      navigator.serviceWorker
        .register("/service-worker.js")
        .then((registration) => {
          console.log("[SW] Service Worker registered:", registration.scope);

          // Handle updates
          registration.addEventListener("updatefound", () => {
            const newWorker = registration.installing;
            if (newWorker) {
              newWorker.addEventListener("statechange", () => {
                if (
                  newWorker.state === "installed" &&
                  navigator.serviceWorker.controller
                ) {
                  // New version available
                  console.log("[SW] New version available");
                  // Optionally show update notification here
                }
              });
            }
          });
        })
        .catch((error) => {
          console.error("[SW] Service Worker registration failed:", error);
        });
    }
  }, []);

  return null;
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <ToastProvider>
        <QueryProvider>
          <AuthProvider>
            <AccessibilityProvider>
              <LocaleProvider>
                <WebSocketProvider>
                  <PerformanceMonitoring />
                  <ServiceWorkerRegistration />
                  {children}
                </WebSocketProvider>
              </LocaleProvider>
            </AccessibilityProvider>
          </AuthProvider>
        </QueryProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
