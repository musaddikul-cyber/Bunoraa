
"use client";

import * as React from "react";
import Image from "next/image";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useForm, useWatch } from "react-hook-form";
import { useToast } from "@/components/ui/ToastProvider";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { formatMoney } from "@/lib/checkout";
import { useAuthContext } from "@/components/providers/AuthProvider";
import {
  usePreorderCategories,
  usePreorderCategory,
  usePreorderTemplates,
  usePreorderEstimate,
  useCreatePreorder,
} from "@/components/preorders/usePreorderData";
import { FileDropzone } from "@/components/preorders/FileDropzone";
import type { PreorderCategory, PreorderOption } from "@/lib/types";
import { apiFetch } from "@/lib/api";

const STORAGE_KEY = "bunoraa-preorder-draft-v2";
const stepOrder = ["category", "customize", "details", "review"] as const;

type StepKey = (typeof stepOrder)[number];

type PreorderDraft = {
  category_id: string;
  category_slug: string;
  template_id?: string;
  title: string;
  description: string;
  quantity: number;
  options: Record<string, unknown>;
  full_name: string;
  email: string;
  phone?: string;
  special_instructions?: string;
  is_gift: boolean;
  gift_wrap: boolean;
  gift_message?: string;
  is_rush_order: boolean;
  requested_delivery_date?: string;
  customer_notes?: string;
  shipping_first_name?: string;
  shipping_last_name?: string;
  shipping_address_line_1?: string;
  shipping_address_line_2?: string;
  shipping_city?: string;
  shipping_state?: string;
  shipping_postal_code?: string;
  shipping_country?: string;
};

const emptyDraft: PreorderDraft = {
  category_id: "",
  category_slug: "",
  template_id: "",
  title: "",
  description: "",
  quantity: 1,
  options: {},
  full_name: "",
  email: "",
  phone: "",
  special_instructions: "",
  is_gift: false,
  gift_wrap: false,
  gift_message: "",
  is_rush_order: false,
  requested_delivery_date: "",
  customer_notes: "",
  shipping_first_name: "",
  shipping_last_name: "",
  shipping_address_line_1: "",
  shipping_address_line_2: "",
  shipping_city: "",
  shipping_state: "",
  shipping_postal_code: "",
  shipping_country: "",
};

const parseStoredDraft = () => {
  if (typeof window === "undefined") return emptyDraft;
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (!saved) return emptyDraft;
  try {
    return { ...emptyDraft, ...(JSON.parse(saved) as PreorderDraft) };
  } catch {
    return emptyDraft;
  }
};

const buildOptionPayload = (
  options: Record<string, unknown>,
  category?: PreorderCategory | null
) => {
  if (!category?.options?.length) return {};
  const payload: Record<string, unknown> = {};
  category.options.forEach((option) => {
    const value = options[option.id];
    if (option.option_type === "file") return;
    if (option.option_type === "multiselect") {
      if (Array.isArray(value) && value.length) payload[option.id] = value;
      return;
    }
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) payload[option.id] = trimmed;
      return;
    }
    if (typeof value === "number") {
      payload[option.id] = value;
      return;
    }
    if (typeof value === "boolean") {
      payload[option.id] = value;
    }
  });
  return payload;
};
export default function PreorderCreateStepPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const { push } = useToast();
  const auth = useAuthContext();
  const stepParam = Number(params?.step || 1);
  const stepIndex = Number.isFinite(stepParam)
    ? Math.min(Math.max(stepParam - 1, 0), stepOrder.length - 1)
    : 0;
  const currentStep = stepOrder[stepIndex] || "category";

  const [defaults] = React.useState<PreorderDraft>(parseStoredDraft);
  const form = useForm<PreorderDraft>({ defaultValues: defaults });
  const watched = useWatch({ control: form.control });

  const categoriesQuery = usePreorderCategories();
  const selectedCategoryId = watched?.category_id || "";
  const selectedCategorySlug = watched?.category_slug || "";
  const categorySlug =
    selectedCategorySlug ||
    categoriesQuery.data?.find((cat) => cat.id === selectedCategoryId)?.slug ||
    "";

  const categoryQuery = usePreorderCategory(categorySlug);
  const templatesQuery = usePreorderTemplates(categorySlug);
  const estimateMutation = usePreorderEstimate();
  const createPreorder = useCreatePreorder();

  const [designFiles, setDesignFiles] = React.useState<File[]>([]);
  const [referenceFiles, setReferenceFiles] = React.useState<File[]>([]);
  const [optionFiles, setOptionFiles] = React.useState<Record<string, File[]>>({});
  const [includeShipping, setIncludeShipping] = React.useState(false);
  const [estimate, setEstimate] = React.useState<typeof estimateMutation.data | null>(null);

  const selectedCategory =
    categoryQuery.data ||
    categoriesQuery.data?.find((cat) => cat.id === selectedCategoryId) ||
    null;
  const templates = templatesQuery.data || [];

  const estimatePayload = React.useMemo(() => {
    if (!selectedCategory || !watched) return null;
    return {
      category_id: selectedCategory.id,
      quantity: watched.quantity || selectedCategory.min_quantity || 1,
      options: buildOptionPayload(watched.options || {}, selectedCategory),
      is_rush_order: watched.is_rush_order || false,
    };
  }, [selectedCategory, watched]);

  React.useEffect(() => {
    const initialCategory = searchParams.get("category");
    if (!initialCategory || selectedCategoryId) return;
    const match = categoriesQuery.data?.find(
      (cat) => cat.slug === initialCategory
    );
    if (match) {
      form.setValue("category_id", match.id);
      form.setValue("category_slug", match.slug);
    }
  }, [searchParams, categoriesQuery.data, form, selectedCategoryId]);

  React.useEffect(() => {
    if (!auth.profileQuery.data) return;
    if (form.getValues("full_name") || form.getValues("email")) return;
    const profile = auth.profileQuery.data;
    form.setValue(
      "full_name",
      `${profile.first_name || ""} ${profile.last_name || ""}`.trim()
    );
    form.setValue("email", profile.email || "");
    form.setValue("phone", profile.phone || "");
  }, [auth.profileQuery.data, form]);

  React.useEffect(() => {
    if (!watched) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(watched));
  }, [watched]);

  React.useEffect(() => {
    if (!estimatePayload) return;
    estimateMutation.mutate(estimatePayload, {
      onSuccess: (data) => setEstimate(data),
      onError: () => setEstimate(null),
    });
  }, [estimatePayload, estimateMutation]);

  const goToStep = (next: StepKey) => {
    const nextIndex = stepOrder.indexOf(next);
    router.push(`/preorders/create/${nextIndex + 1}/`);
  };

  const validateStep = (stepKey: StepKey) => {
    const values = form.getValues();
    if (stepKey === "category") {
      if (!values.category_id) {
        form.setError("category_id", {
          message: "Select a category to continue.",
        });
        return false;
      }
    }
    if (stepKey === "customize") {
      if (!values.title.trim()) {
        form.setError("title", { message: "Add a title for your preorder." });
        return false;
      }
      if (!values.description.trim()) {
        form.setError("description", { message: "Describe what you need." });
        return false;
      }
      if (!values.quantity || values.quantity < 1) {
        form.setError("quantity", { message: "Enter a valid quantity." });
        return false;
      }
      selectedCategory?.options?.forEach((option) => {
        if (!option.is_required) return;
        if (option.option_type === "file") {
          const files = optionFiles[option.id] || [];
          if (!files.length) {
            form.setError(`options.${option.id}` as const, {
              message: `${option.name} is required.`,
            });
          }
          return;
        }
        const value = values.options?.[option.id];
        if (
          value === undefined ||
          value === null ||
          value === "" ||
          (Array.isArray(value) && value.length === 0)
        ) {
          form.setError(`options.${option.id}` as const, {
            message: `${option.name} is required.`,
          });
        }
      });
    }
    if (stepKey === "details") {
      if (!values.full_name.trim()) {
        form.setError("full_name", { message: "Enter your full name." });
        return false;
      }
      if (!values.email.trim()) {
        form.setError("email", { message: "Enter your email address." });
        return false;
      }
    }
    return true;
  };

  const handleContinue = async () => {
    const stepKey = currentStep;
    if (!validateStep(stepKey)) return;
    const nextIndex = Math.min(stepIndex + 1, stepOrder.length - 1);
    goToStep(stepOrder[nextIndex]);
  };

  const handleBack = () => {
    const prevIndex = Math.max(stepIndex - 1, 0);
    goToStep(stepOrder[prevIndex]);
  };

  const handleSubmit = async () => {
    const values = form.getValues();
    if (!validateStep("details")) return;
    if (!values.category_id) {
      push("Select a category first.", "error");
      goToStep("category");
      return;
    }

    const payload: Record<string, unknown> = {
      category: values.category_id,
      title: values.title,
      description: values.description,
      quantity: values.quantity,
      full_name: values.full_name,
      email: values.email,
      phone: values.phone || "",
      options: buildOptionPayload(values.options || {}, selectedCategory),
      special_instructions: values.special_instructions || "",
      is_gift: values.is_gift,
      gift_wrap: values.is_gift ? values.gift_wrap : false,
      gift_message: values.is_gift ? values.gift_message : "",
      is_rush_order: selectedCategory?.allow_rush_order
        ? values.is_rush_order
        : false,
      requested_delivery_date: values.requested_delivery_date || null,
      customer_notes: values.customer_notes || "",
      submit: true,
    };

    if (includeShipping) {
      Object.assign(payload, {
        shipping_first_name: values.shipping_first_name || "",
        shipping_last_name: values.shipping_last_name || "",
        shipping_address_line_1: values.shipping_address_line_1 || "",
        shipping_address_line_2: values.shipping_address_line_2 || "",
        shipping_city: values.shipping_city || "",
        shipping_state: values.shipping_state || "",
        shipping_postal_code: values.shipping_postal_code || "",
        shipping_country: values.shipping_country || "",
      });
    }

    try {
      const created = await createPreorder.mutateAsync(payload);
      const preorderNumber = created.preorder_number;
      if (preorderNumber && auth.hasToken) {
        for (const file of designFiles) {
          const formData = new FormData();
          formData.append("file", file);
          await apiFetch(`/preorders/orders/${preorderNumber}/upload_design/`, {
            method: "POST",
            body: formData,
          });
        }
        for (const file of referenceFiles) {
          const formData = new FormData();
          formData.append("file", file);
          await apiFetch(`/preorders/orders/${preorderNumber}/upload_reference/`, {
            method: "POST",
            body: formData,
          });
        }
        for (const [optionId, files] of Object.entries(optionFiles)) {
          for (const file of files) {
            const formData = new FormData();
            formData.append("option_id", optionId);
            formData.append("file", file);
            await apiFetch(
              `/preorders/orders/${preorderNumber}/upload-option-file/`,
              {
                method: "POST",
                body: formData,
              }
            );
          }
        }
      }
      window.localStorage.removeItem(STORAGE_KEY);
      if (preorderNumber) {
        router.push(`/preorders/success/${preorderNumber}/`);
      } else {
        router.push("/preorders/");
      }
    } catch (error) {
      push(
        error instanceof Error ? error.message : "Could not submit preorder.",
        "error"
      );
    }
  };

  const renderOptionField = (option: PreorderOption) => {
    const error = (form.formState.errors.options as Record<
      string,
      { message?: string }
    > | undefined)?.[option.id];
    const commonClass =
      "mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm";
    const description = option.help_text || option.description;

    switch (option.option_type) {
      case "textarea":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <textarea
              rows={4}
              className={commonClass}
              placeholder={option.placeholder || ""}
              {...form.register(`options.${option.id}` as const)}
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "number":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <input
              type="number"
              className={commonClass}
              {...form.register(`options.${option.id}` as const)}
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "select":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <select
              className={commonClass}
              {...form.register(`options.${option.id}` as const)}
            >
              <option value="">Select {option.name}</option>
              {option.choices?.map((choice) => (
                <option key={choice.id} value={choice.id}>
                  {choice.display_name}
                  {choice.price_modifier ? ` (+${choice.price_modifier})` : ""}
                </option>
              ))}
            </select>
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "multiselect":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <select
              className={commonClass}
              multiple
              value={
                (form.getValues(`options.${option.id}` as const) as string[]) ||
                []
              }
              onChange={(event) => {
                const selected = Array.from(event.target.selectedOptions).map(
                  (opt) => opt.value
                );
                form.setValue(`options.${option.id}` as const, selected);
              }}
            >
              {option.choices?.map((choice) => (
                <option key={choice.id} value={choice.id}>
                  {choice.display_name}
                  {choice.price_modifier ? ` (+${choice.price_modifier})` : ""}
                </option>
              ))}
            </select>
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "checkbox":
        return (
          <label key={option.id} className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-1"
              {...form.register(`options.${option.id}` as const)}
            />
            <span>
              {option.name}
              {description ? (
                <span className="block text-xs text-foreground/60">
                  {description}
                </span>
              ) : null}
            </span>
          </label>
        );
      case "color":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <input
              type="color"
              className="h-12 w-full rounded-lg border border-border bg-card px-2 py-1"
              {...form.register(`options.${option.id}` as const)}
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "date":
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <input
              type="date"
              className={commonClass}
              {...form.register(`options.${option.id}` as const)}
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      case "file":
        return (
          <div key={option.id} className="space-y-1">
            <FileDropzone
              label={option.name}
              description={
                description || "Upload reference files for this option."
              }
              value={optionFiles[option.id] || []}
              multiple={false}
              maxFiles={1}
              onChange={(files) =>
                setOptionFiles((prev) => ({ ...prev, [option.id]: files }))
              }
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
      default:
        return (
          <div key={option.id} className="space-y-1">
            <label className="text-sm font-medium">
              {option.name}
              {option.is_required ? " *" : ""}
            </label>
            {description ? (
              <p className="text-xs text-foreground/60">{description}</p>
            ) : null}
            <input
              className={commonClass}
              placeholder={option.placeholder || ""}
              {...form.register(`options.${option.id}` as const)}
            />
            {error?.message ? (
              <p className="text-xs text-rose-500">{error.message}</p>
            ) : null}
          </div>
        );
    }
  };
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
        <div className="mb-8 space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
            Custom preorder
          </p>
          <h1 className="text-3xl font-semibold">Create a preorder request</h1>
          <p className="text-sm text-foreground/60">
            Share your requirements and we will send a quote with production
            timeline.
          </p>
        </div>

        <div className="mb-6 flex items-center gap-3 overflow-x-auto pb-2 text-xs uppercase tracking-[0.2em] text-foreground/60">
          {stepOrder.map((step, index) => (
            <div
              key={step}
              className={cn(
                "rounded-full border px-4 py-1",
                index === stepIndex
                  ? "border-primary text-primary"
                  : "border-border"
              )}
            >
              {index + 1}. {step}
            </div>
          ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <Card variant="bordered" className="space-y-6">
            {currentStep === "category" ? (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold">Choose a category</h2>
                  <p className="text-sm text-foreground/60">
                    Select the closest match to customize your preorder.
                  </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  {categoriesQuery.data?.map((category) => {
                    const selected = selectedCategoryId === category.id;
                    return (
                      <button
                        key={category.id}
                        type="button"
                        className={cn(
                          "rounded-2xl border p-4 text-left transition",
                          selected
                            ? "border-primary bg-primary/10"
                            : "border-border bg-card hover:bg-muted/40"
                        )}
                        onClick={() => {
                          form.setValue("category_id", category.id);
                          form.setValue("category_slug", category.slug);
                          form.setValue("options", {});
                        }}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-sm font-semibold">
                              {category.name}
                            </p>
                            <p className="text-xs text-foreground/60">
                              {category.description}
                            </p>
                          </div>
                          {category.image ? (
                            <div className="relative h-14 w-14 overflow-hidden rounded-xl bg-muted">
                              <Image
                                src={category.image}
                                alt={category.name}
                                fill
                                className="object-cover"
                              />
                            </div>
                          ) : null}
                        </div>
                        <div className="mt-3 flex flex-wrap gap-3 text-xs text-foreground/70">
                          {category.base_price ? (
                            <span>
                              From {formatMoney(category.base_price, "BDT")}
                            </span>
                          ) : null}
                          {category.deposit_percentage ? (
                            <span>Deposit {category.deposit_percentage}%</span>
                          ) : null}
                          {category.min_production_days ? (
                            <span>
                              {category.min_production_days}-
                              {category.max_production_days} days
                            </span>
                          ) : null}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {selectedCategory ? (
                  <div className="space-y-3 rounded-2xl border border-border bg-muted/30 p-4">
                    <p className="text-sm font-semibold">Recommended templates</p>
                    <div className="grid gap-3 md:grid-cols-2">
                      {templates.length ? (
                        templates.map((template) => (
                          <button
                            key={template.id}
                            type="button"
                            className={cn(
                              "rounded-xl border px-3 py-2 text-left text-sm",
                              watched?.template_id === template.id
                                ? "border-primary bg-primary/10"
                                : "border-border bg-card"
                            )}
                            onClick={() => {
                              form.setValue("template_id", template.id);
                              if (template.name)
                                form.setValue("title", template.name);
                              if (template.description)
                                form.setValue(
                                  "description",
                                  template.description
                                );
                              if (template.default_quantity)
                                form.setValue(
                                  "quantity",
                                  template.default_quantity
                                );
                              if (template.default_options) {
                                form.setValue(
                                  "options",
                                  template.default_options as Record<
                                    string,
                                    unknown
                                  >
                                );
                              }
                            }}
                          >
                            <p className="font-semibold">{template.name}</p>
                            <p className="text-xs text-foreground/60">
                              {template.description}
                            </p>
                          </button>
                        ))
                      ) : (
                        <p className="text-xs text-foreground/60">
                          No templates available.
                        </p>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            {currentStep === "customize" ? (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold">
                    Customization details
                  </h2>
                  <p className="text-sm text-foreground/60">
                    Tell us what you need and pick the options that fit best.
                  </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    Title
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      {...form.register("title")}
                    />
                    {form.formState.errors.title?.message ? (
                      <p className="mt-1 text-xs text-rose-500">
                        {form.formState.errors.title.message}
                      </p>
                    ) : null}
                  </label>
                  <label className="block text-sm">
                    Quantity
                    <input
                      type="number"
                      min={selectedCategory?.min_quantity || 1}
                      max={selectedCategory?.max_quantity || undefined}
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      {...form.register("quantity", { valueAsNumber: true })}
                    />
                    {form.formState.errors.quantity?.message ? (
                      <p className="mt-1 text-xs text-rose-500">
                        {form.formState.errors.quantity.message}
                      </p>
                    ) : null}
                  </label>
                </div>
                <label className="block text-sm">
                  Description
                  <textarea
                    rows={4}
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                    {...form.register("description")}
                  />
                  {form.formState.errors.description?.message ? (
                    <p className="mt-1 text-xs text-rose-500">
                      {form.formState.errors.description.message}
                    </p>
                  ) : null}
                </label>
                {selectedCategory?.options?.length ? (
                  <div className="space-y-4">
                    <p className="text-sm font-semibold">
                      Customization options
                    </p>
                    {selectedCategory.options.map((option) =>
                      renderOptionField(option)
                    )}
                  </div>
                ) : null}
              </div>
            ) : null}

            {currentStep === "details" ? (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold">Contact and delivery</h2>
                  <p className="text-sm text-foreground/60">
                    We will use this info to send your quote and timeline.
                  </p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block text-sm">
                    Full name
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      {...form.register("full_name")}
                    />
                    {form.formState.errors.full_name?.message ? (
                      <p className="mt-1 text-xs text-rose-500">
                        {form.formState.errors.full_name.message}
                      </p>
                    ) : null}
                  </label>
                  <label className="block text-sm">
                    Email
                    <input
                      type="email"
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      {...form.register("email")}
                    />
                    {form.formState.errors.email?.message ? (
                      <p className="mt-1 text-xs text-rose-500">
                        {form.formState.errors.email.message}
                      </p>
                    ) : null}
                  </label>
                  <label className="block text-sm md:col-span-2">
                    Phone
                    <input
                      className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      {...form.register("phone")}
                    />
                  </label>
                </div>

                {selectedCategory?.allow_rush_order ? (
                  <label className="flex items-start gap-2 text-sm">
                    <input type="checkbox" {...form.register("is_rush_order")} />
                    <span>
                      Rush production
                      <span className="block text-xs text-foreground/60">
                        Adds {selectedCategory.rush_order_fee_percentage}% rush
                        fee.
                      </span>
                    </span>
                  </label>
                ) : null}

                <label className="block text-sm">
                  Requested delivery date (optional)
                  <input
                    type="date"
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                    {...form.register("requested_delivery_date")}
                  />
                </label>

                <label className="flex items-start gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={includeShipping}
                    onChange={(event) => setIncludeShipping(event.target.checked)}
                  />
                  <span>Add shipping details</span>
                </label>

                {includeShipping ? (
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="block text-sm">
                      Shipping first name
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_first_name")}
                      />
                    </label>
                    <label className="block text-sm">
                      Shipping last name
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_last_name")}
                      />
                    </label>
                    <label className="block text-sm md:col-span-2">
                      Address line 1
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_address_line_1")}
                      />
                    </label>
                    <label className="block text-sm md:col-span-2">
                      Address line 2
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_address_line_2")}
                      />
                    </label>
                    <label className="block text-sm">
                      City
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_city")}
                      />
                    </label>
                    <label className="block text-sm">
                      State / Province
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_state")}
                      />
                    </label>
                    <label className="block text-sm">
                      Postal code
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_postal_code")}
                      />
                    </label>
                    <label className="block text-sm">
                      Country
                      <input
                        className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                        {...form.register("shipping_country")}
                      />
                    </label>
                  </div>
                ) : null}

                <div className="space-y-3">
                  <label className="flex items-start gap-2 text-sm">
                    <input type="checkbox" {...form.register("is_gift")} />
                    <span>Mark as a gift</span>
                  </label>
                  {watched.is_gift ? (
                    <div className="space-y-2">
                      <label className="flex items-start gap-2 text-sm">
                        <input type="checkbox" {...form.register("gift_wrap")} />
                        <span>Include gift wrap</span>
                      </label>
                      <label className="block text-sm">
                        Gift message
                        <textarea
                          rows={3}
                          className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                          {...form.register("gift_message")}
                        />
                      </label>
                    </div>
                  ) : null}
                </div>

                <label className="block text-sm">
                  Special instructions
                  <textarea
                    rows={3}
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                    {...form.register("special_instructions")}
                  />
                </label>
                <label className="block text-sm">
                  Notes for the team
                  <textarea
                    rows={3}
                    className="mt-2 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                    {...form.register("customer_notes")}
                  />
                </label>

                {selectedCategory?.requires_design ? (
                  <div className="grid gap-4 md:grid-cols-2">
                    <FileDropzone
                      label="Design files"
                      description="Upload logos, sketches, or artwork for production."
                      accept=".pdf,.png,.jpg,.jpeg,.ai,.psd,.svg,.eps,.cdr,.zip,.rar"
                      multiple
                      maxFiles={5}
                      value={designFiles}
                      onChange={setDesignFiles}
                    />
                    <FileDropzone
                      label="Reference images"
                      description="Optional inspiration or reference files."
                      accept=".png,.jpg,.jpeg,.pdf,.zip"
                      multiple
                      maxFiles={5}
                      value={referenceFiles}
                      onChange={setReferenceFiles}
                    />
                  </div>
                ) : null}
                {!auth.hasToken && (designFiles.length || referenceFiles.length) ? (
                  <p className="text-xs text-foreground/60">
                    Sign in to upload files. Guest submissions will save files
                    after login.
                  </p>
                ) : null}
              </div>
            ) : null}

            {currentStep === "review" ? (
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-semibold">Review and submit</h2>
                  <p className="text-sm text-foreground/60">
                    Confirm your details before sending the preorder request.
                  </p>
                </div>
                <div className="space-y-3 text-sm text-foreground/70">
                  <p>
                    <strong>Category:</strong> {selectedCategory?.name || "Not selected"}
                  </p>
                  <p>
                    <strong>Title:</strong> {watched.title}
                  </p>
                  <p>
                    <strong>Quantity:</strong> {watched.quantity}
                  </p>
                  <p>
                    <strong>Contact:</strong> {watched.full_name} ({watched.email})
                  </p>
                  {estimate ? (
                    <div className="rounded-xl border border-border bg-muted/30 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                        Estimate
                      </p>
                      <div className="mt-2 space-y-1">
                        <p>Subtotal: {formatMoney(estimate.subtotal, estimate.currency)}</p>
                        <p>Rush fee: {formatMoney(estimate.rush_fee, estimate.currency)}</p>
                        <p className="font-semibold">
                          Total: {formatMoney(estimate.total, estimate.currency)}
                        </p>
                        <p className="text-xs text-foreground/60">
                          Deposit required: {formatMoney(estimate.deposit_required, estimate.currency)}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-foreground/60">
                      Estimated pricing will appear here.
                    </p>
                  )}
                </div>
                <div className="rounded-xl border border-border bg-muted/30 p-4 text-xs text-foreground/60">
                  We will review your request and send a quote with production
                  timeline and next steps. No payment details are collected
                  until you approve the quote.
                </div>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-3">
              <Button variant="secondary" onClick={handleBack} disabled={stepIndex === 0}>
                Back
              </Button>
              {currentStep !== "review" ? (
                <Button onClick={handleContinue}>Continue</Button>
              ) : (
                <Button
                  variant="primary-gradient"
                  onClick={handleSubmit}
                  disabled={createPreorder.isPending}
                >
                  {createPreorder.isPending ? "Submitting..." : "Submit preorder"}
                </Button>
              )}
            </div>
          </Card>

          <Card variant="glass" className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
                Summary
              </p>
              <h3 className="text-lg font-semibold">Preorder snapshot</h3>
            </div>
            <div className="space-y-2 text-sm text-foreground/70">
              <p>Category: {selectedCategory?.name || "Not selected"}</p>
              <p>Quantity: {watched.quantity || 0}</p>
              {estimate ? (
                <>
                  <p>Estimated total: {formatMoney(estimate.total, estimate.currency)}</p>
                  <p>Deposit required: {formatMoney(estimate.deposit_required, estimate.currency)}</p>
                </>
              ) : null}
              {selectedCategory?.min_production_days ? (
                <p>
                  Lead time: {selectedCategory.min_production_days}-
                  {selectedCategory.max_production_days} days
                </p>
              ) : null}
            </div>
            <div className="rounded-xl border border-border bg-card p-3 text-xs text-foreground/60">
              {selectedCategory?.requires_approval
                ? "All preorders are reviewed by our team before production."
                : "We will confirm your preorder details via email."}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
