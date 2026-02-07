"use client";

import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import Link from "next/link";

export default function DeleteAccountPage() {
  return (
    <Card variant="bordered" className="space-y-4 p-6">
      <h1 className="text-2xl font-semibold">Delete account</h1>
      <p className="text-sm text-foreground/70">
        Manage account deletion and export requests from the privacy center.
      </p>
      <Button asChild variant="secondary">
        <Link href="/account/privacy/">Go to privacy center</Link>
      </Button>
    </Card>
  );
}
