"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAuth } from "@/components/auth/useAuth";

const schema = z.object({
  email: z.string().email("Enter a valid email"),
  password: z.string().min(1, "Password is required"),
  remember: z.boolean().optional(),
});

type FormValues = z.infer<typeof schema>;

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuth();
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { remember: true },
  });

  const nextUrl = searchParams.get("next") || "/account/profile/";

  const onSubmit = async (values: FormValues) => {
    await login.mutateAsync({
      email: values.email,
      password: values.password,
      remember: Boolean(values.remember),
    });
    router.push(nextUrl);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-md px-6 py-20">
        <Card variant="bordered" className="space-y-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Account
            </p>
            <h1 className="text-2xl font-semibold">Sign in</h1>
          </div>

          <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
            <label className="block text-sm">
              Email
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="email"
                {...form.register("email")}
              />
            </label>

            <label className="block text-sm">
              Password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                {...form.register("password")}
              />
            </label>

            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" {...form.register("remember")} />
              Remember me
            </label>

            {login.isError ? (
              <p className="text-sm text-red-500">
                {login.error instanceof Error ? login.error.message : "Login failed."}
              </p>
            ) : null}

            <Button type="submit" className="w-full" disabled={login.isPending}>
              {login.isPending ? "Signing in..." : "Sign in"}
            </Button>
          </form>

          <div className="flex items-center justify-between text-sm text-foreground/70">
            <Link className="text-primary" href="/account/forgot-password/">
              Forgot password?
            </Link>
            <Link className="text-primary" href="/account/register/">
              Create account
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
