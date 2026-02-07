"use client";

import { useEffect, useRef } from "react";
import { useRouter, useParams } from "next/navigation";
import { useAddresses } from "@/components/account/useAddresses";
import { Card } from "@/components/ui/Card";

export default function SetDefaultAddressPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;
  const { updateAddress } = useAddresses();
  const didRunRef = useRef(false);

  useEffect(() => {
    if (!id || didRunRef.current) return;
    didRunRef.current = true;
    updateAddress
      .mutateAsync({ id, payload: { is_default: true } })
      .then(() => router.push("/account/addresses/"));
  }, [id, router, updateAddress]);

  return (
    <Card variant="bordered" className="p-6 text-sm text-foreground/70">
      Setting default address...
    </Card>
  );
}
