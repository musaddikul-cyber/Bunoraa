"use client";

import * as React from "react";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/Button";

type Status = "idle" | "loading" | "success" | "error";

export function FooterNewsletter() {
  const [email, setEmail] = React.useState("");
  const [status, setStatus] = React.useState<Status>("idle");
  const [message, setMessage] = React.useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) {
      setStatus("error");
      setMessage("Please enter a valid email address.");
      return;
    }
    setStatus("loading");
    setMessage(null);
    try {
      await apiFetch("/pages/subscribers/", {
        method: "POST",
        body: { email: trimmed },
      });
      setStatus("success");
      setMessage("Thanks for subscribing. Watch your inbox for updates.");
      setEmail("");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Unable to subscribe right now.");
    }
  };

  return (
    <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
      <p className="text-sm font-semibold">Join the Bunoraa circle</p>
      <p className="mt-1 text-xs text-foreground/70">
        Monthly drops, artisan stories, and early access invites.
      </p>
      <form className="mt-3 flex flex-col gap-2 sm:flex-row" onSubmit={handleSubmit}>
        <input
          className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
          type="email"
          placeholder="Email address"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          aria-label="Email address"
          required
        />
        <Button
          type="submit"
          variant="primary-gradient"
          className="shrink-0"
          disabled={status === "loading"}
        >
          {status === "loading" ? "Joining..." : "Join"}
        </Button>
      </form>
      {message ? (
        <p className="mt-2 text-xs text-foreground/70" aria-live="polite">
          {message}
        </p>
      ) : null}
    </div>
  );
}
