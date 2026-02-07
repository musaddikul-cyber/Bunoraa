"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useAuth } from "@/components/auth/useAuth";

const schema = z
  .object({
    email: z.string().email("Enter a valid email"),
    password: z.string().min(8, "Password must be at least 8 characters"),
    password_confirm: z.string().min(8, "Confirm your password"),
    first_name: z.string().optional(),
    last_name: z.string().optional(),
    phone: z.string().optional(),
  })
  .refine((data) => data.password === data.password_confirm, {
    message: "Passwords do not match",
    path: ["password_confirm"],
  });

type FormValues = z.infer<typeof schema>;

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();
  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      email: "",
      password: "",
      password_confirm: "",
      first_name: "",
      last_name: "",
      phone: "",
    },
  });

  const onSubmit = async (values: FormValues) => {
    await register.mutateAsync(values);
    router.push("/account/login/");
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-md px-6 py-20">
        <Card variant="bordered" className="space-y-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
              Account
            </p>
            <h1 className="text-2xl font-semibold">Create account</h1>
          </div>

          <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
            <label className="block text-sm">
              First name
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                {...form.register("first_name")}
              />
            </label>
            <label className="block text-sm">
              Last name
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                {...form.register("last_name")}
              />
            </label>
            <label className="block text-sm">
              Phone
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                {...form.register("phone")}
              />
            </label>
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
              <span className="mt-2 block text-xs text-foreground/60">
                Use a long, unique passphrase (12+ characters recommended).
              </span>
            </label>
            <label className="block text-sm">
              Confirm password
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
                type="password"
                {...form.register("password_confirm")}
              />
            </label>

            {register.isError ? (
              <p className="text-sm text-red-500">
                {register.error instanceof Error
                  ? register.error.message
                  : "Registration failed."}
              </p>
            ) : null}

            <Button type="submit" className="w-full" disabled={register.isPending}>
              {register.isPending ? "Creating..." : "Create account"}
            </Button>
          </form>

          <p className="text-sm text-foreground/70">
            Already have an account?{" "}
            <Link className="text-primary" href="/account/login/">
              Sign in
            </Link>
          </p>
        </Card>
      </div>
    </div>
  );
}
