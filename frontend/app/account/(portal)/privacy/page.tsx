"use client";

import * as React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { useExports } from "@/components/account/useExports";
import type { AccountDeletionStatus } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

async function fetchDeletionStatus() {
  const response = await apiFetch<AccountDeletionStatus | null>(
    "/accounts/delete/status/"
  );
  return response.data;
}

export default function PrivacyPage() {
  const queryClient = useQueryClient();
  const { exportsQuery, requestExport } = useExports();
  const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");
  const deletionQuery = useQuery({
    queryKey: ["account", "deletion"],
    queryFn: fetchDeletionStatus,
  });
  const [reason, setReason] = React.useState("");

  const requestDeletion = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<AccountDeletionStatus>(
        "/accounts/delete/",
        { method: "POST", body: { reason } }
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["account", "deletion"] });
    },
  });

  const cancelDeletion = useMutation({
    mutationFn: async () => {
      const response = await apiFetch<AccountDeletionStatus>(
        "/accounts/delete/cancel/",
        { method: "POST" }
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["account", "deletion"] });
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-3xl font-semibold">Privacy & data</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Export your data or schedule account deletion.
        </p>
      </div>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Data export</h2>
        <p className="text-sm text-foreground/70">
          Generate a downloadable archive of your profile, orders, and
          preferences.
        </p>
        <Button
          onClick={() => requestExport.mutate()}
          disabled={requestExport.isPending}
        >
          {requestExport.isPending ? "Requesting..." : "Request export"}
        </Button>
        {exportsQuery.data?.length ? (
          <div className="space-y-2 text-sm">
            {exportsQuery.data.map((job) => (
              <div
                key={job.id}
                className="flex flex-col gap-2 rounded-lg border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
              >
                <div>
                  <p className="font-semibold capitalize">{job.status}</p>
                  <p className="text-xs text-foreground/60">
                    Requested: {job.requested_at}
                  </p>
                </div>
                {job.file ? (
                  <a
                    className="text-primary"
                    href={`${apiBase}/accounts/export/${job.id}/download/`}
                  >
                    Download
                  </a>
                ) : (
                  <span className="text-xs text-foreground/60">
                    Preparing file
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : null}
      </Card>

      <Card variant="bordered" className="space-y-4">
        <h2 className="text-lg font-semibold">Delete account</h2>
        <p className="text-sm text-foreground/70">
          Schedule account deletion. You can cancel before the grace period
          ends.
        </p>
        <label className="block text-sm">
          Reason (optional)
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
          />
        </label>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            onClick={() => requestDeletion.mutate()}
            disabled={requestDeletion.isPending}
          >
            {requestDeletion.isPending ? "Scheduling..." : "Request deletion"}
          </Button>
          <Button
            variant="ghost"
            onClick={() => cancelDeletion.mutate()}
            disabled={cancelDeletion.isPending}
          >
            Cancel deletion
          </Button>
        </div>
        {deletionQuery.data ? (
          <div className="rounded-lg border border-border bg-muted p-3 text-sm text-foreground/70">
            Status: {deletionQuery.data.status || "pending"}
            {deletionQuery.data.scheduled_for ? (
              <p>Scheduled for: {deletionQuery.data.scheduled_for}</p>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-foreground/60">No deletion request.</p>
        )}
      </Card>
    </div>
  );
}
