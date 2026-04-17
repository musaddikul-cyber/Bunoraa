"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatAddressLine } from "@/lib/address";
import type { Address, Country } from "@/lib/types";

const schema = z.object({
  email: z.string().email("Enter a valid email"),
  shipping_first_name: z.string().min(1, "First name is required"),
  shipping_last_name: z.string().min(1, "Last name is required"),
  shipping_phone: z.string().min(1, "Phone number is required"),
  shipping_address_line_1: z.string().min(1, "Address line 1 is required"),
  shipping_address_line_2: z.string().optional(),
  shipping_city: z.string().min(1, "City is required"),
  shipping_state: z.string().optional(),
  shipping_postal_code: z.string().min(1, "Postal code is required"),
  shipping_country: z.string().min(1, "Country is required"),
  saved_shipping_address_id: z.string().uuid().nullable().optional(),
  save_address: z.boolean().optional(),
});

export type CheckoutInfoFormValues = z.infer<typeof schema>;

type CheckoutInfoStepProps = {
  defaultValues: Partial<CheckoutInfoFormValues>;
  countries: Country[];
  savedAddresses: Address[];
  defaultSelectedAddressId?: string | null;
  allowSaveAddress?: boolean;
  onSubmit: (values: CheckoutInfoFormValues) => Promise<void>;
  onSavedAddressSelectionChange?: (values: CheckoutInfoFormValues) => void;
  isSubmitting?: boolean;
  isAutoSavingSelection?: boolean;
};

export function CheckoutInfoStep({
  defaultValues,
  countries,
  savedAddresses,
  defaultSelectedAddressId,
  allowSaveAddress = false,
  onSubmit,
  onSavedAddressSelectionChange,
  isSubmitting,
  isAutoSavingSelection,
}: CheckoutInfoStepProps) {
  const NEW_ADDRESS_ID = "new";
  const MAX_SAVED_ADDRESSES = 4;
  const form = useForm<CheckoutInfoFormValues>({
    resolver: zodResolver(schema),
    defaultValues,
  });

  const [selectedAddress, setSelectedAddress] = React.useState<string>("");
  const shouldPersistSelectedAddressRef = React.useRef(false);

  React.useEffect(() => {
    form.reset(defaultValues);
  }, [defaultValues, form]);

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

  const normalize = React.useCallback((value?: string | null) => {
    return (value || "").trim().toLowerCase();
  }, []);

  const findMatchedSavedAddress = React.useCallback(() => {
    const shippingLine1 = normalize(defaultValues.shipping_address_line_1);
    const shippingCity = normalize(defaultValues.shipping_city);
    const shippingPostal = normalize(defaultValues.shipping_postal_code);
    const shippingState = normalize(defaultValues.shipping_state);
    const shippingCountry = normalize(
      resolveCountryName(defaultValues.shipping_country || "")
    );

    if (!shippingLine1 || !shippingCity || !shippingPostal || !shippingCountry) {
      return null;
    }

    return (
      savedAddresses.find((address) => {
        const addressCountry = normalize(resolveCountryName(address.country));
        return (
          normalize(address.address_line_1) === shippingLine1 &&
          normalize(address.city) === shippingCity &&
          normalize(address.postal_code) === shippingPostal &&
          normalize(address.state) === shippingState &&
          addressCountry === shippingCountry
        );
      }) || null
    );
  }, [
    defaultValues.shipping_address_line_1,
    defaultValues.shipping_city,
    defaultValues.shipping_postal_code,
    defaultValues.shipping_state,
    defaultValues.shipping_country,
    normalize,
    resolveCountryName,
    savedAddresses,
  ]);

  React.useEffect(() => {
    if (!savedAddresses.length) {
      setSelectedAddress("");
      return;
    }

    const hasExistingSelection = selectedAddress
      ? savedAddresses.some((address) => address.id === selectedAddress) ||
        selectedAddress === NEW_ADDRESS_ID
      : false;
    if (hasExistingSelection) return;

    // Priority 1: Use explicit saved_address_id from session (returns UUID)
    if (
      defaultSelectedAddressId &&
      savedAddresses.some((address) => address.id === defaultSelectedAddressId)
    ) {
      setSelectedAddress(defaultSelectedAddressId);
      return;
    }

    // Priority 2: Try to match address fields to a saved address
    const matchedSavedAddressId = findMatchedSavedAddress()?.id;
    if (matchedSavedAddressId) {
      setSelectedAddress(matchedSavedAddressId);
      return;
    }

    // Priority 3: If there are address fields but no match, use new address
    const hasSessionAddress =
      Boolean(normalize(defaultValues.shipping_address_line_1)) ||
      Boolean(normalize(defaultValues.shipping_city)) ||
      Boolean(normalize(defaultValues.shipping_postal_code));

    if (hasSessionAddress) {
      setSelectedAddress(NEW_ADDRESS_ID);
      return;
    }

    const defaultAddress =
      savedAddresses.find((address) => address.is_default) || savedAddresses[0];
    setSelectedAddress(defaultAddress?.id || NEW_ADDRESS_ID);
  }, [
    defaultSelectedAddressId,
    defaultValues.shipping_address_line_1,
    defaultValues.shipping_city,
    defaultValues.shipping_country,
    defaultValues.shipping_postal_code,
    findMatchedSavedAddress,
    normalize,
    savedAddresses,
    selectedAddress,
    NEW_ADDRESS_ID,
  ]);

  React.useEffect(() => {
    if (!selectedAddress || selectedAddress === NEW_ADDRESS_ID) return;
    const address = savedAddresses.find((item) => item.id === selectedAddress);
    if (!address) return;
    form.setValue("shipping_address_line_1", address.address_line_1 || "");
    form.setValue("shipping_address_line_2", address.address_line_2 || "");
    form.setValue("shipping_city", address.city || "");
    form.setValue("shipping_state", address.state || "");
    form.setValue("shipping_postal_code", address.postal_code || "");
    form.setValue("shipping_country", resolveCountryName(address.country));
    form.setValue("save_address", false);

    if (shouldPersistSelectedAddressRef.current && onSavedAddressSelectionChange) {
      shouldPersistSelectedAddressRef.current = false;
      const currentValues = form.getValues();
      const payload: CheckoutInfoFormValues = {
        ...currentValues,
        shipping_address_line_1: address.address_line_1 || "",
        shipping_address_line_2: address.address_line_2 || "",
        shipping_city: address.city || "",
        shipping_state: address.state || "",
        shipping_postal_code: address.postal_code || "",
        shipping_country: resolveCountryName(address.country),
        save_address: false,
        saved_shipping_address_id: selectedAddress,
      };
      const parsed = schema.safeParse(payload);
      if (parsed.success) {
        onSavedAddressSelectionChange({
          ...parsed.data,
          save_address: false,
          saved_shipping_address_id: selectedAddress,
        });
      }
    }
  }, [
    selectedAddress,
    savedAddresses,
    form,
    resolveCountryName,
    NEW_ADDRESS_ID,
    onSavedAddressSelectionChange,
  ]);

  const clearAddressFields = React.useCallback(() => {
    form.setValue("shipping_address_line_1", "");
    form.setValue("shipping_address_line_2", "");
    form.setValue("shipping_city", "");
    form.setValue("shipping_state", "");
    form.setValue("shipping_postal_code", "");
    form.setValue("shipping_country", "");
  }, [form]);

  const handleSelectAddress = React.useCallback(
    (id: string) => {
      shouldPersistSelectedAddressRef.current = Boolean(
        id && id !== NEW_ADDRESS_ID && id !== selectedAddress
      );
      setSelectedAddress(id);
      if (!id || id === NEW_ADDRESS_ID) {
        clearAddressFields();
        form.setValue("saved_shipping_address_id", null);
      } else {
        form.setValue("save_address", false);
        form.setValue("saved_shipping_address_id", id);
      }
    },
    [clearAddressFields, NEW_ADDRESS_ID, form, selectedAddress]
  );

  const sortedCountries = React.useMemo(() => {
    return [...countries].sort((a, b) => a.name.localeCompare(b.name));
  }, [countries]);

  const renderError = (name: keyof CheckoutInfoFormValues) => {
    const error = form.formState.errors[name];
    if (!error) return null;
    return (
      <p className="mt-1 text-xs text-rose-500" id={`${name}-error`}>
        {error.message}
      </p>
    );
  };

  const handleSubmit = form.handleSubmit(async (values) => {
    const isSavedAddressSelected = Boolean(
      savedAddresses.length && selectedAddress && selectedAddress !== NEW_ADDRESS_ID
    );
    const isNewAddress = !savedAddresses.length || !isSavedAddressSelected;
    const canSaveMore = savedAddresses.length < MAX_SAVED_ADDRESSES;
    const shouldSave =
      allowSaveAddress && isNewAddress && canSaveMore && Boolean(values.save_address);
    await onSubmit({
      ...values,
      save_address: shouldSave,
      saved_shipping_address_id: isSavedAddressSelected ? selectedAddress : null,
    });
  });

  return (
    <Card variant="bordered" className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
          Step 1
        </p>
        <h2 className="text-xl font-semibold">Shipping information</h2>
        <p className="text-sm text-foreground/60">
          Tell us where you want your order delivered.
        </p>
      </div>

      <form className="grid gap-4" onSubmit={handleSubmit}>
        <label className="block text-sm">
          Email
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            type="email"
            autoComplete="email"
            {...form.register("email")}
            aria-invalid={Boolean(form.formState.errors.email)}
            aria-describedby={form.formState.errors.email ? "email-error" : undefined}
          />
          {renderError("email")}
        </label>

        <div className="grid gap-4 md:grid-cols-2">
          <label className="block text-sm">
            First name
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
              autoComplete="shipping given-name"
              {...form.register("shipping_first_name")}
              aria-invalid={Boolean(form.formState.errors.shipping_first_name)}
              aria-describedby={
                form.formState.errors.shipping_first_name
                  ? "shipping_first_name-error"
                  : undefined
              }
            />
            {renderError("shipping_first_name")}
          </label>
          <label className="block text-sm">
            Last name
            <input
              className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
              autoComplete="shipping family-name"
              {...form.register("shipping_last_name")}
              aria-invalid={Boolean(form.formState.errors.shipping_last_name)}
              aria-describedby={
                form.formState.errors.shipping_last_name
                  ? "shipping_last_name-error"
                  : undefined
              }
            />
            {renderError("shipping_last_name")}
          </label>
        </div>

        <label className="block text-sm">
          Phone
          <input
            className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
            type="tel"
            autoComplete="shipping tel"
            inputMode="tel"
            {...form.register("shipping_phone")}
            aria-invalid={Boolean(form.formState.errors.shipping_phone)}
            aria-describedby={
              form.formState.errors.shipping_phone
                ? "shipping_phone-error"
                : undefined
            }
          />
          {renderError("shipping_phone")}
        </label>

        {savedAddresses.length ? (
          <div className="mt-6 space-y-4 border-t border-border pt-6">
            <div>
              <p className="text-sm font-semibold">Saved addresses</p>
              <p className="text-xs text-foreground/60">
                Select a saved address or enter a new one.
              </p>
              {isAutoSavingSelection ? (
                <p className="text-xs text-foreground/60">
                  Saving selected address...
                </p>
              ) : null}
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {savedAddresses.map((address) => {
                const isSelected = selectedAddress === address.id;
                return (
                  <label
                    key={address.id}
                    className={`flex cursor-pointer flex-col gap-2 rounded-xl border px-4 py-3 text-sm ${
                      isSelected
                        ? "border-primary bg-primary/10"
                        : "border-border bg-card hover:bg-muted"
                    }`}
                  >
                    <input
                      type="radio"
                      className="sr-only"
                      name="saved_address"
                      value={address.id}
                      checked={isSelected}
                      onChange={() => handleSelectAddress(address.id)}
                    />
                    <div className="flex items-center justify-between">
                      <p className="font-semibold">
                        {address.full_name || "Saved address"}
                      </p>
                      {address.is_default ? (
                        <span className="text-xs text-primary">Default</span>
                      ) : null}
                    </div>
                    {address.phone ? (
                      <p className="text-xs text-foreground/60">{address.phone}</p>
                    ) : null}
                    <p className="text-xs text-foreground/70">
                      {formatAddressLine(address, { resolveCountryName })}
                    </p>
                  </label>
                );
              })}
              <label
                className={`flex cursor-pointer flex-col gap-2 rounded-xl border px-4 py-3 text-sm ${
                  selectedAddress === NEW_ADDRESS_ID
                    ? "border-primary bg-primary/10"
                    : "border-border bg-card hover:bg-muted"
                }`}
              >
                <input
                  type="radio"
                  className="sr-only"
                  name="saved_address"
                  value={NEW_ADDRESS_ID}
                  checked={selectedAddress === NEW_ADDRESS_ID}
                  onChange={() => handleSelectAddress(NEW_ADDRESS_ID)}
                />
                <p className="font-semibold">Use a new address</p>
                <p className="text-xs text-foreground/60">
                  Enter a different shipping address.
                </p>
              </label>
            </div>
          </div>
        ) : null}

        {!savedAddresses.length || selectedAddress === NEW_ADDRESS_ID ? (
          <>
            <div className="mt-2 border-t border-border" role="separator" aria-hidden="true" />
            <label className="block text-sm">
              Address line 1
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="shipping address-line1"
                {...form.register("shipping_address_line_1")}
                aria-invalid={Boolean(form.formState.errors.shipping_address_line_1)}
                aria-describedby={
                  form.formState.errors.shipping_address_line_1
                    ? "shipping_address_line_1-error"
                    : undefined
                }
              />
              {renderError("shipping_address_line_1")}
            </label>

            <label className="block text-sm">
              Address line 2 (optional)
              <input
                className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                autoComplete="shipping address-line2"
                {...form.register("shipping_address_line_2")}
              />
            </label>

            <div className="grid gap-4 md:grid-cols-2">
              <label className="block text-sm">
                City
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                  autoComplete="shipping address-level2"
                  {...form.register("shipping_city")}
                  aria-invalid={Boolean(form.formState.errors.shipping_city)}
                  aria-describedby={
                    form.formState.errors.shipping_city
                      ? "shipping_city-error"
                      : undefined
                  }
                />
                {renderError("shipping_city")}
              </label>
              <label className="block text-sm">
                State / Province
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                  autoComplete="shipping address-level1"
                  {...form.register("shipping_state")}
                />
              </label>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <label className="block text-sm">
                Postal code
                <input
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                  autoComplete="shipping postal-code"
                  inputMode="text"
                  {...form.register("shipping_postal_code")}
                  aria-invalid={Boolean(form.formState.errors.shipping_postal_code)}
                  aria-describedby={
                    form.formState.errors.shipping_postal_code
                      ? "shipping_postal_code-error"
                      : undefined
                  }
                />
                {renderError("shipping_postal_code")}
              </label>
              <label className="block text-sm">
                Country
                <select
                  className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                  autoComplete="shipping country-name"
                  {...form.register("shipping_country")}
                  aria-invalid={Boolean(form.formState.errors.shipping_country)}
                  aria-describedby={
                    form.formState.errors.shipping_country
                      ? "shipping_country-error"
                      : undefined
                  }
                >
                  <option value="">Select country</option>
                  {sortedCountries.map((country) => (
                    <option key={country.code} value={country.name}>
                      {country.flag_emoji ? `${country.flag_emoji} ` : ""}
                      {country.name}
                    </option>
                  ))}
                </select>
                {renderError("shipping_country")}
              </label>
            </div>

            {allowSaveAddress ? (
              <label className="flex items-start gap-2 rounded-lg border border-border bg-card px-3 py-3 text-sm">
                <input
                  type="checkbox"
                  className="mt-1"
                  disabled={savedAddresses.length >= MAX_SAVED_ADDRESSES}
                  {...form.register("save_address")}
                />
                <span>
                  Save this address to my account.
                  <span className="mt-1 block text-xs text-foreground/60">
                    {savedAddresses.length >= MAX_SAVED_ADDRESSES
                      ? `You can save up to ${MAX_SAVED_ADDRESSES} addresses.`
                      : "You can reuse it for future checkouts."}
                  </span>
                </span>
              </label>
            ) : null}
          </>
        ) : null}

        <Button type="submit" className="w-full" disabled={isSubmitting}>
          {isSubmitting ? "Saving..." : "Continue to shipping"}
        </Button>
      </form>
    </Card>
  );
}
