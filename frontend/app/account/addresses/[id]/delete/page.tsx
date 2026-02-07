"use client";

import { useRouter, useParams } from "next/navigation";
import { AuthGate } from "@/components/auth/AuthGate";
import { useAddresses } from "@/components/account/useAddresses";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default function DeleteAddressPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;
  const { deleteAddress } = useAddresses();

  const handleDelete = async () => {
    await deleteAddress.mutateAsync(id);
    router.push("/account/addresses/");
  };

  return (
    <AuthGate>
      <div className="mx-auto w-full max-w-lg px-6 py-12">
        <Card variant="bordered" className="space-y-4 p-6">
          <h1 className="text-2xl font-semibold">Delete address</h1>
          <p className="text-sm text-foreground/70">
            Are you sure you want to delete this address? This action cannot be undone.
          </p>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => router.back()}>
              Cancel
            </Button>
            <Button variant="ghost" onClick={handleDelete} disabled={deleteAddress.isPending}>
              {deleteAddress.isPending ? "Deleting..." : "Delete"}
            </Button>
          </div>
        </Card>
      </div>
    </AuthGate>
  );
}
