import type { Address } from "@/lib/types";

type AddressLike = Partial<
  Pick<
    Address,
    | "address_line_1"
    | "address_line_2"
    | "city"
    | "state"
    | "postal_code"
    | "country"
    | "full_address"
  >
>;

type FormatAddressOptions = {
  countryName?: string | null;
  resolveCountryName?: (value?: string | null) => string | null | undefined;
};

const normalize = (value?: string | null) =>
  typeof value === "string" ? value.trim() : "";

export function formatAddressLine(
  address?: AddressLike | null,
  options?: FormatAddressOptions
) {
  if (!address) return "";

  const addressLine1 = normalize(address.address_line_1);
  const addressLine2 = normalize(address.address_line_2);
  const city = normalize(address.city);
  const postal = normalize(address.postal_code);
  const state = normalize(address.state);

  const resolvedCountry = options?.countryName
    ? normalize(options.countryName)
    : options?.resolveCountryName
      ? normalize(options.resolveCountryName(address.country) || "")
      : normalize(address.country);

  const cityPostal = city && postal ? `${city}-${postal}` : city || postal;
  const parts: string[] = [];

  if (addressLine2) parts.push(addressLine2);
  if (addressLine1) parts.push(addressLine1);
  if (cityPostal) parts.push(cityPostal);
  if (state) parts.push(state);
  if (resolvedCountry) parts.push(resolvedCountry);

  if (parts.length) return parts.join(", ");

  const fallback = normalize(address.full_address);
  return fallback || addressLine1 || "";
}
