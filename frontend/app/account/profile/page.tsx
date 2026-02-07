"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { AuthGate } from "@/components/auth/AuthGate";
import { useAuth } from "@/components/auth/useAuth";

export default function ProfilePage() {
  const { profileQuery, updateProfile, logout } = useAuth();
  const profile = profileQuery.data;
  const [form, setForm] = React.useState({
    first_name: "",
    last_name: "",
    phone: "",
    newsletter_subscribed: false,
  });

  React.useEffect(() => {
    if (profile) {
      setForm({
        first_name: profile.first_name || "",
        last_name: profile.last_name || "",
        phone: profile.phone || "",
        newsletter_subscribed: Boolean(profile.newsletter_subscribed),
      });
    }
  }, [profile]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    await updateProfile.mutateAsync(form);
  };

  return (
    <AuthGate>
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto w-full max-w-3xl px-6 py-16">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
                Account
              </p>
              <h1 className="text-2xl font-semibold">Profile</h1>
            </div>
            <Button variant="ghost" onClick={logout}>
              Sign out
            </Button>
          </div>

          <Card variant="bordered" className="space-y-4">
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block text-sm">
                  First name
                  <input
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                    name="first_name"
                    value={form.first_name}
                    onChange={handleChange}
                  />
                </label>
                <label className="block text-sm">
                  Last name
                  <input
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                    name="last_name"
                    value={form.last_name}
                    onChange={handleChange}
                  />
                </label>
              </div>
              <label className="block text-sm">
                Phone
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                  name="phone"
                  value={form.phone}
                  onChange={handleChange}
                />
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  name="newsletter_subscribed"
                  checked={form.newsletter_subscribed}
                  onChange={handleChange}
                />
                Subscribe to newsletter
              </label>

              <Button type="submit" disabled={updateProfile.isPending}>
                {updateProfile.isPending ? "Saving..." : "Save changes"}
              </Button>
            </form>
          </Card>
        </div>
      </div>
    </AuthGate>
  );
}
