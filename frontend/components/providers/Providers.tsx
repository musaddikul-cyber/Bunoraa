"use client";

import * as React from "react";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { ThemeProvider } from "@/components/theme/ThemeProvider";
import { AuthProvider } from "@/components/providers/AuthProvider";
import { LocaleProvider } from "@/components/providers/LocaleProvider";
import { WebSocketProvider } from "@/components/providers/WebSocketProvider";
import { ToastProvider } from "@/components/ui/ToastProvider";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <ToastProvider>
        <QueryProvider>
          <AuthProvider>
            <LocaleProvider>
              <WebSocketProvider>{children}</WebSocketProvider>
            </LocaleProvider>
          </AuthProvider>
        </QueryProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
