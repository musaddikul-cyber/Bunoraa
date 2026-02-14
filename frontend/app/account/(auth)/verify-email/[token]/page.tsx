"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";

export default function VerifyEmailPage() {
  const params = useParams();
  const token = params?.token as string;
  const [status, setStatus] = useState<"pending" | "success" | "error">(
    "pending"
  );

  useEffect(() => {
    if (!token) return;
    apiFetch("/accounts/email/verify/", {
      method: "POST",
      body: { token },
    })
      .then(() => setStatus("success"))
      .catch(() => setStatus("error"));
  }, [token]);

  return (
    <div className="mx-auto w-full max-w-md px-4 sm:px-6 py-20">
      <Card variant="bordered" className="p-6 text-sm text-foreground/70">
        {status === "pending" && "Verifying your email..."}
        {status === "success" && "Email verified successfully. You can sign in now."}
        {status === "error" && "Verification failed. The token may be invalid or expired."}
      </Card>
    </div>
  );
}
