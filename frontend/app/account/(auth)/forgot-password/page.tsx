"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

const schema = z.object({
  email: z.string().email("Enter a valid email"),
});

type FormValues = z.infer<typeof schema>;

export default function ForgotPasswordPage() {
  const [sent, setSent] = useState(false);
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
  });

  const onSubmit = async (values: FormValues) => {
    await apiFetch("/accounts/password/reset/request/", {
      method: "POST",
      body: values,
    });
    setSent(true);
  };

  return (
    <div className="mx-auto w-full max-w-md px-4 sm:px-6 py-20">
      <Card variant="bordered" className="space-y-6">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Account
          </p>
          <h1 className="text-2xl font-semibold">Reset password</h1>
        </div>

        {sent ? (
          <p className="text-sm text-foreground/70">
            If an account exists with that email, a reset link has been sent.
          </p>
        ) : (
          <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
            <label className="block text-sm">
              Email
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="email"
                {...form.register("email")}
              />
            </label>
            <Button type="submit" className="w-full">
              Send reset link
            </Button>
          </form>
        )}
      </Card>
    </div>
  );
}
