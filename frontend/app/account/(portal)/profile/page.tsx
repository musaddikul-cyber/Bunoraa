"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAuth } from "@/components/auth/useAuth";
import { apiFetch } from "@/lib/api";

export default function ProfilePage() {
  const { profileQuery, updateProfile, logout } = useAuth();
  const profile = profileQuery.data;
  const [form, setForm] = React.useState({
    first_name: "",
    last_name: "",
    phone: "",
    date_of_birth: "",
    newsletter_subscribed: false,
  });
  const [avatarUploading, setAvatarUploading] = React.useState(false);
  const [verificationMessage, setVerificationMessage] = React.useState<string | null>(null);
  const [verificationSending, setVerificationSending] = React.useState(false);

  React.useEffect(() => {
    if (profile) {
      setForm({
        first_name: profile.first_name || "",
        last_name: profile.last_name || "",
        phone: profile.phone || "",
        date_of_birth: profile.date_of_birth || "",
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

  const handleAvatarChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setAvatarUploading(true);
    try {
      const formData = new FormData();
      formData.append("avatar", file);
      await apiFetch("/accounts/profile/avatar/", {
        method: "POST",
        body: formData,
      });
      await profileQuery.refetch();
    } finally {
      setAvatarUploading(false);
    }
  };

  const handleResendVerification = async () => {
    setVerificationMessage(null);
    setVerificationSending(true);
    try {
      const response = await apiFetch("/accounts/email/resend/", { method: "POST" });
      setVerificationMessage(response.message || "Verification email sent.");
    } catch (error) {
      setVerificationMessage(
        error instanceof Error ? error.message : "Failed to send verification email."
      );
    } finally {
      setVerificationSending(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
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

      <Card variant="bordered" className="space-y-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="h-16 w-16 overflow-hidden rounded-full bg-muted">
              {profile?.avatar ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={profile.avatar}
                  alt={profile.full_name || "Profile"}
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-xl font-semibold text-foreground/50">
                  {profile?.first_name?.[0] || "U"}
                </div>
              )}
            </div>
            <div>
              <p className="text-sm text-foreground/60">Email</p>
              <p className="font-semibold">{profile?.email || "-"}</p>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-foreground/60">
                <span>
                  {profile?.is_verified ? "Verified account" : "Email not verified"}
                </span>
                {!profile?.is_verified ? (
                  <button
                    type="button"
                    className="text-xs font-semibold text-primary hover:underline"
                    onClick={handleResendVerification}
                    disabled={verificationSending}
                  >
                    {verificationSending ? "Sending..." : "Send verification email"}
                  </button>
                ) : null}
              </div>
              {verificationMessage ? (
                <p className="text-xs text-foreground/60">{verificationMessage}</p>
              ) : null}
            </div>
          </div>
          <label className="inline-flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleAvatarChange}
            />
            <span className="rounded-lg border border-border bg-card px-3 py-2 text-sm">
              {avatarUploading ? "Uploading..." : "Change photo"}
            </span>
          </label>
        </div>

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
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block text-sm">
              Phone
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                name="phone"
                value={form.phone}
                onChange={handleChange}
              />
            </label>
            <label className="block text-sm">
              Date of birth
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                name="date_of_birth"
                type="date"
                value={form.date_of_birth}
                onChange={handleChange}
              />
            </label>
          </div>
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
  );
}
