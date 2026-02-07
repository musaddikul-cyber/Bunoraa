"use client";

import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";

export default function DeleteAccountPage() {
  return (
    <AuthGate>
      <div className="mx-auto w-full max-w-md px-6 py-20">
        <Card variant="bordered" className="space-y-4 p-6">
          <h1 className="text-2xl font-semibold">Delete account</h1>
          <p className="text-sm text-foreground/70">
            Account deletion is not available via API yet. Please contact support
            to complete this request.
          </p>
        </Card>
      </div>
    </AuthGate>
  );
}
