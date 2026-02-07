"use client";

import * as React from "react";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

export default function ContactPage() {
  const [form, setForm] = React.useState({
    name: "",
    email: "",
    phone: "",
    subject: "",
    message: "",
  });
  const [status, setStatus] = React.useState<string | null>(null);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm((prev) => ({ ...prev, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setStatus(null);
    try {
      await apiFetch("/pages/contact/", {
        method: "POST",
        body: form,
      });
      setStatus("Message sent. We will get back to you soon.");
      setForm({ name: "", email: "", phone: "", subject: "", message: "" });
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to send message.");
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-6 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Contact
          </p>
          <h1 className="text-3xl font-semibold">Get in touch</h1>
        </div>
        <Card variant="bordered">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="grid gap-4 sm:grid-cols-2">
              <input
                className="rounded-lg border border-border bg-card px-3 py-2"
                placeholder="Your name"
                name="name"
                value={form.name}
                onChange={handleChange}
                required
              />
              <input
                className="rounded-lg border border-border bg-card px-3 py-2"
                placeholder="Email"
                name="email"
                type="email"
                value={form.email}
                onChange={handleChange}
                required
              />
              <input
                className="rounded-lg border border-border bg-card px-3 py-2"
                placeholder="Phone"
                name="phone"
                value={form.phone}
                onChange={handleChange}
              />
              <input
                className="rounded-lg border border-border bg-card px-3 py-2"
                placeholder="Subject"
                name="subject"
                value={form.subject}
                onChange={handleChange}
                required
              />
            </div>
            <textarea
              className="min-h-[160px] w-full rounded-lg border border-border bg-card px-3 py-2"
              placeholder="Message"
              name="message"
              value={form.message}
              onChange={handleChange}
              required
            />
            {status ? (
              <p className="text-sm text-foreground/70">{status}</p>
            ) : null}
            <Button type="submit" variant="primary-gradient">
              Send message
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
}
