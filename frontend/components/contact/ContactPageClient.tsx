"use client";

import * as React from "react";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { JsonLd } from "@/components/seo/JsonLd";

export function ContactPageClient() {
  const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000").replace(
    /\/$/,
    ""
  );
  const contactSchema = {
    "@context": "https://schema.org",
    "@type": "ContactPage",
    name: "Contact Bunoraa",
    url: `${siteUrl}/contact/`,
  };
  const [form, setForm] = React.useState({
    name: "",
    email: "",
    phone: "",
    subject: "",
    message: "",
  });
  const [status, setStatus] = React.useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm((prev) => ({ ...prev, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (isSubmitting) return;
    setStatus(null);
    setIsSubmitting(true);
    try {
      await apiFetch("/pages/contact/", {
        method: "POST",
        body: form,
      });
      setStatus({ tone: "success", message: "Message sent. We will get back to you soon." });
      setForm({ name: "", email: "", phone: "", subject: "", message: "" });
    } catch (error) {
      setStatus({
        tone: "error",
        message: error instanceof Error ? error.message : "Failed to send message.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-3 sm:px-5 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Contact
          </p>
          <h1 className="text-3xl font-semibold">Get in touch</h1>
        </div>
        <Card variant="bordered">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="grid gap-1.5 text-sm font-medium text-foreground/80">
                <span>Your name</span>
                <input
                  id="contact-name"
                  className="h-11 rounded-lg border border-border bg-card px-3"
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  autoComplete="name"
                  required
                />
              </label>
              <label className="grid gap-1.5 text-sm font-medium text-foreground/80">
                <span>Email address</span>
                <input
                  id="contact-email"
                  className="h-11 rounded-lg border border-border bg-card px-3"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={handleChange}
                  autoComplete="email"
                  required
                />
              </label>
              <label className="grid gap-1.5 text-sm font-medium text-foreground/80">
                <span>Phone (optional)</span>
                <input
                  id="contact-phone"
                  className="h-11 rounded-lg border border-border bg-card px-3"
                  name="phone"
                  value={form.phone}
                  onChange={handleChange}
                  autoComplete="tel"
                  inputMode="tel"
                />
              </label>
              <label className="grid gap-1.5 text-sm font-medium text-foreground/80">
                <span>Subject</span>
                <input
                  id="contact-subject"
                  className="h-11 rounded-lg border border-border bg-card px-3"
                  name="subject"
                  value={form.subject}
                  onChange={handleChange}
                  autoComplete="off"
                  required
                />
              </label>
            </div>
            <label className="grid gap-1.5 text-sm font-medium text-foreground/80">
              <span>Message</span>
              <textarea
                id="contact-message"
                className="min-h-[160px] w-full rounded-lg border border-border bg-card px-3 py-2"
                name="message"
                value={form.message}
                onChange={handleChange}
                required
              />
            </label>
            {status ? (
              <p
                className={`text-sm ${status.tone === "error" ? "text-error-500" : "text-foreground/70"}`}
                aria-live="polite"
              >
                {status.message}
              </p>
            ) : null}
            <Button
              type="submit"
              variant="primary-gradient"
              className="w-full sm:w-auto"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Sending..." : "Send message"}
            </Button>
          </form>
        </Card>
      </div>
      <JsonLd data={contactSchema} />
    </div>
  );
}
