"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type { Country } from "@/lib/types";

async function fetchCountries() {
  const response = await apiFetch<Country[]>("/i18n/countries/");
  return response.data;
}

export function useCountries(options?: { enabled?: boolean }) {
  const countriesQuery = useQuery({
    queryKey: ["i18n", "countries"],
    queryFn: fetchCountries,
    enabled: options?.enabled ?? true,
  });

  const countries = React.useMemo(() => {
    const list = countriesQuery.data || [];
    return [...list].sort((a, b) => a.name.localeCompare(b.name));
  }, [countriesQuery.data]);

  return { countriesQuery, countries };
}
