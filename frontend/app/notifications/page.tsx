"use client";

import * as React from "react";
import { useNotifications } from "@/components/notifications/useNotifications";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function NotificationsPage() {
  const { notificationsQuery, markAllRead, markRead } = useNotifications();
  const [selected, setSelected] = React.useState<string[]>([]);

  const notifications = notificationsQuery.data || [];

  const toggle = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  return (
    <AuthGate title="Notifications" description="Sign in to view notifications.">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Notifications
          </p>
          <h1 className="text-3xl font-semibold">Your updates</h1>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            onClick={() => markAllRead.mutate()}
            disabled={markAllRead.isPending}
          >
            Mark all read
          </Button>
          <Button
            variant="ghost"
            onClick={() => markRead.mutate(selected)}
            disabled={selected.length === 0 || markRead.isPending}
          >
            Mark selected
          </Button>
        </div>
      </div>

      {notificationsQuery.isLoading ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Loading notifications...
        </Card>
      ) : notifications.length === 0 ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          You are all caught up.
        </Card>
      ) : (
        <div className="space-y-4">
          {notifications.map((note) => (
            <Card
              key={note.id}
              variant="bordered"
              className={`flex items-start justify-between gap-4 p-4 ${
                note.is_read ? "opacity-70" : ""
              }`}
            >
              <div>
                <h2 className="text-base font-semibold">
                  {note.title || "Notification"}
                </h2>
                <p className="text-sm text-foreground/70">{note.message}</p>
              </div>
              <input
                type="checkbox"
                checked={selected.includes(note.id)}
                onChange={() => toggle(note.id)}
              />
            </Card>
          ))}
        </div>
      )}
      </div>
    </AuthGate>
  );
}
