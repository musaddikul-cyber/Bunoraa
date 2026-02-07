"use client";

import Link from "next/link";
import { useAddresses } from "@/components/account/useAddresses";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatAddressLine } from "@/lib/address";

export default function AddressesPage() {
  const { addressesQuery, deleteAddress } = useAddresses();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            Account
          </p>
          <h1 className="text-3xl font-semibold">Addresses</h1>
        </div>
        <Button asChild variant="primary-gradient">
          <Link href="/account/addresses/add/">Add address</Link>
        </Button>
      </div>

      {addressesQuery.isLoading ? (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          Loading addresses...
        </Card>
      ) : addressesQuery.data?.length ? (
        <div className="grid gap-4 md:grid-cols-2">
          {addressesQuery.data.map((address) => (
            <Card key={address.id} variant="bordered" className="space-y-2 p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">
                  {address.full_name || "Address"}
                </h2>
                {address.is_default ? (
                  <span className="text-xs uppercase tracking-[0.2em] text-primary">
                    Default
                  </span>
                ) : null}
              </div>
              <p className="text-sm text-foreground/70">
                {formatAddressLine(address)}
              </p>
              <div className="flex flex-wrap gap-2">
                <Button asChild size="sm" variant="secondary">
                  <Link href={`/account/addresses/${address.id}/edit/`}>Edit</Link>
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => deleteAddress.mutate(address.id)}
                >
                  Delete
                </Button>
                <Button asChild size="sm" variant="ghost">
                  <Link href={`/account/addresses/${address.id}/set-default/`}>
                    Set default
                  </Link>
                </Button>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card variant="bordered" className="p-6 text-sm text-foreground/70">
          No addresses yet.
        </Card>
      )}
    </div>
  );
}
