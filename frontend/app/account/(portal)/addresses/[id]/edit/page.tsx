"use client";

import * as React from "react";
import { useRouter, useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiFetch } from "@/lib/api";
import { useAddresses } from "@/components/account/useAddresses";
import { useAuthContext } from "@/components/providers/AuthProvider";
import type { Address } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useCountries } from "@/components/i18n/useCountries";

const schema = z.object({
  address_line_1: z.string().min(1, "Address line 1 is required"),
  address_line_2: z.string().optional(),
  city: z.string().min(1, "City is required"),
  state: z.string().optional(),
  postal_code: z.string().min(1, "Postal code is required"),
  country: z.string().min(1, "Country is required"),
  is_default: z.boolean().optional(),
});

type FormValues = z.infer<typeof schema>;

export default function EditAddressPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;
  const { updateAddress } = useAddresses();
  const { profileQuery } = useAuthContext();
  const { countries, countriesQuery } = useCountries();
  const profile = profileQuery.data;
  const profileName =
    profile?.full_name ||
    [profile?.first_name, profile?.last_name].filter(Boolean).join(" ");
  const profilePhone = profile?.phone || "";

  const resolveCountryName = React.useCallback(
    (value?: string | null) => {
      if (!value) return "";
      const trimmed = value.trim();
      if (!trimmed) return "";
      const byCode = countries.find(
        (country) => country.code.toLowerCase() === trimmed.toLowerCase()
      );
      if (byCode) return byCode.name;
      const byName = countries.find(
        (country) => country.name.toLowerCase() === trimmed.toLowerCase()
      );
      return byName?.name || trimmed;
    },
    [countries]
  );

  const addressQuery = useQuery({
    queryKey: ["addresses", id],
    queryFn: async () => {
      const response = await apiFetch<Address>(`/accounts/addresses/${id}/`);
      return response.data;
    },
    enabled: Boolean(id),
  });

  const resolvedCountry = React.useMemo(
    () => resolveCountryName(addressQuery.data?.country),
    [addressQuery.data?.country, resolveCountryName]
  );

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    values: {
      address_line_1: addressQuery.data?.address_line_1 || "",
      address_line_2: addressQuery.data?.address_line_2 || "",
      city: addressQuery.data?.city || "",
      state: addressQuery.data?.state || "",
      postal_code: addressQuery.data?.postal_code || "",
      country: resolvedCountry,
      is_default: addressQuery.data?.is_default || false,
    },
  });

  const onSubmit = async (values: FormValues) => {
    const name = addressQuery.data?.full_name || profileName;
    const phone = addressQuery.data?.phone || profilePhone;
    if (!name || !phone) return;
    await updateAddress.mutateAsync({
      id,
      payload: {
        ...values,
        full_name: name,
        phone,
      },
    });
    router.push("/account/addresses/");
  };

  return (
    <Card variant="bordered" className="p-6">
      <h1 className="text-2xl font-semibold">Edit address</h1>
      {!addressQuery.isLoading &&
      !(addressQuery.data?.full_name && addressQuery.data?.phone) &&
      !(profileName && profilePhone) ? (
        <div className="mt-4 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-200">
          Please add your full name and phone number in your profile before
          saving an address.
        </div>
      ) : null}
      <form className="mt-6 space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
        <label className="block text-sm">
          Address line 1
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("address_line_1")}
          />
        </label>
        <label className="block text-sm">
          Address line 2
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("address_line_2")}
          />
        </label>
        <label className="block text-sm">
          City
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("city")}
          />
        </label>
        <label className="block text-sm">
          State
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("state")}
          />
        </label>
        <label className="block text-sm">
          Postal code
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("postal_code")}
          />
        </label>
        <label className="block text-sm">
          Country
          <select
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2"
            {...form.register("country", {
              setValueAs: (value) => resolveCountryName(value),
            })}
          >
            <option value="">
              {countriesQuery.isLoading ? "Loading countries..." : "Select country"}
            </option>
            {countries.map((country) => (
              <option key={country.code} value={country.name}>
                {country.flag_emoji ? `${country.flag_emoji} ` : ""}
                {country.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" {...form.register("is_default")} />
          Set as default
        </label>
        <Button
          type="submit"
          className="w-full"
          disabled={
            updateAddress.isPending ||
            (!addressQuery.isLoading &&
              !(addressQuery.data?.full_name && addressQuery.data?.phone) &&
              !(profileName && profilePhone))
          }
        >
          {updateAddress.isPending ? "Saving..." : "Save changes"}
        </Button>
      </form>
    </Card>
  );
}
