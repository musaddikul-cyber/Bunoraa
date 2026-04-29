"use client";

import * as React from "react";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

type ShareState = {
  name: string;
  permission: string;
  expires_days: number;
  password: string;
};

type ShareResult = {
  share_url?: string;
  share_token?: string;
};

export function ShareModal({
  isOpen,
  onClose,
  shareState,
  onShareStateChange,
  shareResult,
  onShare,
  onCopyLink,
}: {
  isOpen: boolean;
  onClose: () => void;
  shareState: ShareState;
  onShareStateChange: (state: ShareState) => void;
  shareResult: ShareResult | null;
  onShare: () => void;
  onCopyLink: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <Card className="relative w-full max-w-md rounded-2xl bg-card p-6 shadow-2xl">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 rounded-full p-1 text-foreground/70 hover:bg-muted hover:text-foreground"
          aria-label="Close dialog"
        >
          <X className="h-5 w-5" />
        </button>

        <h2 className="text-xl font-semibold">Share your bag</h2>
        <p className="mt-1 text-sm text-foreground/70">
          Share your current bag with family, team members, or your personal shopper.
        </p>

        <div className="mt-6 space-y-4">
          <input
            type="text"
            placeholder="Name (optional)"
            value={shareState.name}
            onChange={(event) =>
              onShareStateChange({ ...shareState, name: event.target.value })
            }
            className="h-10 w-full rounded-xl border border-border bg-transparent px-3 text-sm"
          />

          <div className="grid gap-2 sm:grid-cols-2">
            <select
              value={shareState.permission}
              onChange={(event) =>
                onShareStateChange({
                  ...shareState,
                  permission: event.target.value,
                })
              }
              className="h-10 rounded-xl border border-border bg-card px-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
            >
              <option value="view">View only</option>
              <option value="edit">Can edit items</option>
              <option value="purchase">Can purchase</option>
            </select>

            <div className="relative">
              <input
                type="number"
                min={1}
                max={365}
                value={shareState.expires_days}
                onChange={(event) =>
                  onShareStateChange({
                    ...shareState,
                    expires_days: Number(event.target.value || 7),
                  })
                }
                className="no-spin h-10 w-full rounded-xl border border-border bg-card px-3 pr-10 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                placeholder="Expires in days"
                inputMode="numeric"
                pattern="[0-9]*"
              />
              <div className="absolute right-1 top-1 flex h-8 w-8 flex-col overflow-hidden rounded-lg border border-border bg-muted/40">
                <button
                  type="button"
                  className="flex h-4 items-center justify-center border-b border-border/70 text-foreground/70 hover:bg-muted"
                  aria-label="Increase days"
                  onClick={() =>
                    onShareStateChange({
                      ...shareState,
                      expires_days: Math.min(365, shareState.expires_days + 1),
                    })
                  }
                >
                  <ChevronUp aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={2} />
                </button>
                <button
                  type="button"
                  className="flex h-4 items-center justify-center text-foreground/70 hover:bg-muted"
                  aria-label="Decrease days"
                  onClick={() =>
                    onShareStateChange({
                      ...shareState,
                      expires_days: Math.max(1, shareState.expires_days - 1),
                    })
                  }
                >
                  <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={2} />
                </button>
              </div>
            </div>
          </div>

          <input
            type="password"
            placeholder="Password (optional)"
            value={shareState.password}
            onChange={(event) =>
              onShareStateChange({ ...shareState, password: event.target.value })
            }
            className="h-10 w-full rounded-xl border border-border bg-transparent px-3 text-sm"
          />

          {shareResult?.share_url ? (
            <div className="rounded-xl border border-success-500/30 bg-success-500/5 p-3">
              <p className="break-all text-xs text-success-600">{shareResult.share_url}</p>
            </div>
          ) : null}

          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              className="flex-1"
              onClick={onShare}
            >
              Create link
            </Button>
            {shareResult?.share_url ? (
              <Button
                size="sm"
                variant="ghost"
                className="flex-1"
                onClick={onCopyLink}
              >
                Copy link
              </Button>
            ) : null}
          </div>

          <Button
            size="sm"
            variant="ghost"
            className="w-full"
            onClick={onClose}
          >
            Close
          </Button>
        </div>
      </Card>
    </div>
  );
}
