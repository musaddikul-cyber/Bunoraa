"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

const schema = z
  .object({
    current_password: z.string().min(1, "Current password is required"),
    new_password: z.string().min(8, "Password must be at least 8 characters"),
    new_password_confirm: z
      .string()
      .min(8, "Confirm your new password"),
  })
  .refine((data) => data.new_password === data.new_password_confirm, {
    message: "Passwords do not match",
    path: ["new_password_confirm"],
  });

type FormValues = z.infer<typeof schema>;

export default function ChangePasswordPage() {
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
  });

  const onSubmit = async (values: FormValues) => {
    await apiFetch("/accounts/password/change/", {
      method: "POST",
      body: values,
    });
    form.reset();
  };

  return (
    <Card variant="bordered" className="space-y-6">
      <div>
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          Account
        </p>
        <h1 className="text-2xl font-semibold">Change password</h1>
        <p className="mt-2 text-sm text-foreground/70">
          Use a long, unique passphrase that is easy to remember but hard to
          guess.
        </p>
      </div>
      <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
        <label className="block text-sm">
          Current password
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            type="password"
            {...form.register("current_password")}
          />
        </label>
        <label className="block text-sm">
          New password
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            type="password"
            {...form.register("new_password")}
          />
        </label>
        <label className="block text-sm">
          Confirm new password
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            type="password"
            {...form.register("new_password_confirm")}
          />
        </label>
        <Button type="submit" className="w-full">
          Update password
        </Button>
      </form>
    </Card>
  );
}
