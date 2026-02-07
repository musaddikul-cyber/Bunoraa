"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AuthGate } from "@/components/auth/AuthGate";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useToast } from "@/components/ui/ToastProvider";
import { FileDropzone } from "@/components/preorders/FileDropzone";
import { usePreorderDetail, usePreorderActions } from "@/components/preorders/usePreorderData";
import { formatMoney } from "@/lib/checkout";
import type { PreorderQuote } from "@/lib/types";

const formatDate = (value?: string | null) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
};

const formatDateTime = (value?: string | null) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
};

const joinAddress = (...parts: Array<string | null | undefined>) =>
  parts.filter(Boolean).join(", ");

const getErrorMessage = (error: unknown) =>
  error instanceof Error ? error.message : "Something went wrong.";

export default function PreorderDetailPage() {
  const params = useParams();
  const preorderNumberParam = params?.preorder_number;
  const preorderNumber = Array.isArray(preorderNumberParam)
    ? preorderNumberParam[0]
    : preorderNumberParam;
  const { push } = useToast();

  const preorderQuery = usePreorderDetail(preorderNumber);
  const actions = usePreorderActions(preorderNumber);

  const [messageSubject, setMessageSubject] = React.useState("");
  const [messageBody, setMessageBody] = React.useState("");
  const [designFiles, setDesignFiles] = React.useState<File[]>([]);
  const [referenceFiles, setReferenceFiles] = React.useState<File[]>([]);
  const [rejectReasons, setRejectReasons] = React.useState<Record<string, string>>({});

  const preorder = preorderQuery.data;
  const currency = preorder?.currency || "BDT";
  const statusHistory = preorder?.status_history
    ? [...preorder.status_history].reverse()
    : [];

  const handleSendMessage = async () => {
    if (!messageBody.trim()) {
      push("Please write a message.", "error");
      return;
    }
    try {
      await actions.sendMessage.mutateAsync({
        message: messageBody.trim(),
        subject: messageSubject.trim(),
      });
      setMessageBody("");
      setMessageSubject("");
      push("Message sent.", "success");
    } catch (error) {
      push(getErrorMessage(error), "error");
    }
  };

  const handleUpload = async (type: "design" | "reference") => {
    const files = type === "design" ? designFiles : referenceFiles;
    if (!files.length) return;
    try {
      if (type === "design") {
        await Promise.all(files.map((file) => actions.uploadDesign.mutateAsync({ file })));
        setDesignFiles([]);
        push("Design files uploaded.", "success");
      } else {
        await Promise.all(
          files.map((file) => actions.uploadReference.mutateAsync({ file }))
        );
        setReferenceFiles([]);
        push("Reference files uploaded.", "success");
      }
    } catch (error) {
      push(getErrorMessage(error), "error");
    }
  };

  const handleAcceptQuote = async (quote: PreorderQuote) => {
    if (!quote.id) return;
    const confirmed = window.confirm("Accept this quote and proceed to deposit?");
    if (!confirmed) return;
    try {
      await actions.acceptQuote.mutateAsync(quote.id);
      push("Quote accepted.", "success");
    } catch (error) {
      push(getErrorMessage(error), "error");
    }
  };

  const handleRejectQuote = async (quote: PreorderQuote) => {
    if (!quote.id) return;
    const reason = rejectReasons[quote.id] || "";
    const confirmed = window.confirm("Reject this quote?");
    if (!confirmed) return;
    try {
      await actions.rejectQuote.mutateAsync({ quoteId: quote.id, reason });
      push("Quote rejected.", "success");
    } catch (error) {
      push(getErrorMessage(error), "error");
    }
  };

  return (
    <AuthGate
      title="Preorder"
      description="Sign in to view preorder details."
      nextHref={preorderNumber ? `/preorders/order/${preorderNumber}/` : undefined}
    >
      <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6 sm:py-12">
        {preorderQuery.isLoading ? (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Loading preorder...
          </Card>
        ) : preorder ? (
          <div className="space-y-6">
            <Card variant="bordered" className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Preorder
                  </p>
                  <h1 className="text-2xl font-semibold">#{preorder.preorder_number}</h1>
                </div>
                <div className="rounded-full bg-muted px-3 py-1 text-xs">
                  {preorder.status_display || preorder.status}
                </div>
              </div>
              <div className="grid gap-3 text-sm text-foreground/70 md:grid-cols-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Category
                  </p>
                  <p>{preorder.category_name || "Custom"}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Created
                  </p>
                  <p>{formatDate(preorder.created_at)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                    Quantity
                  </p>
                  <p>{preorder.quantity}</p>
                </div>
              </div>
            </Card>

            <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
              <div className="space-y-6">
                <Card variant="bordered" className="space-y-4">
                  <h2 className="text-lg font-semibold">Overview</h2>
                  <div className="grid gap-4 text-sm text-foreground/70 sm:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                        Title
                      </p>
                      <p>{preorder.title}</p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                        Contact
                      </p>
                      <p>{preorder.full_name}</p>
                      <p>{preorder.email}</p>
                      {preorder.phone ? <p>{preorder.phone}</p> : null}
                    </div>
                  </div>
                  <div className="space-y-2 text-sm text-foreground/70">
                    <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                      Description
                    </p>
                    <p>{preorder.description}</p>
                  </div>
                  {preorder.special_instructions ? (
                    <div className="space-y-2 text-sm text-foreground/70">
                      <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                        Special instructions
                      </p>
                      <p>{preorder.special_instructions}</p>
                    </div>
                  ) : null}
                </Card>

                {preorder.option_values?.length ? (
                  <Card variant="bordered" className="space-y-4">
                    <h2 className="text-lg font-semibold">Customization</h2>
                    <div className="grid gap-3 text-sm text-foreground/70 sm:grid-cols-2">
                      {preorder.option_values.map((option) => (
                        <div key={option.id} className="rounded-xl border border-border p-3">
                          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                            {option.option_name}
                          </p>
                          <p className="mt-1 font-medium">
                            {option.display_value || "-"}
                          </p>
                        </div>
                      ))}
                    </div>
                  </Card>
                ) : null}

                <Card variant="bordered" className="space-y-4">
                  <h2 className="text-lg font-semibold">Files</h2>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold">Design files</p>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleUpload("design")}
                          disabled={actions.uploadDesign.isPending || !designFiles.length}
                        >
                          Upload
                        </Button>
                      </div>
                      <FileDropzone
                        label="Add design"
                        description="Upload updated artwork, logos, or sketches."
                        accept=".pdf,.png,.jpg,.jpeg,.ai,.psd,.svg,.eps,.cdr,.zip,.rar"
                        multiple
                        maxFiles={5}
                        value={designFiles}
                        onChange={setDesignFiles}
                      />
                      <div className="space-y-2 text-xs text-foreground/60">
                        {preorder.designs?.length ? (
                          preorder.designs.map((design) => (
                            <a
                              key={design.id}
                              href={design.file || "#"}
                              className="block text-primary"
                              target="_blank"
                              rel="noreferrer"
                            >
                              {design.original_filename || "Design file"}
                            </a>
                          ))
                        ) : (
                          <p>No design files uploaded yet.</p>
                        )}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold">Reference files</p>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleUpload("reference")}
                          disabled={actions.uploadReference.isPending || !referenceFiles.length}
                        >
                          Upload
                        </Button>
                      </div>
                      <FileDropzone
                        label="Add reference"
                        description="Share inspiration or similar products."
                        accept=".pdf,.png,.jpg,.jpeg,.zip"
                        multiple
                        maxFiles={5}
                        value={referenceFiles}
                        onChange={setReferenceFiles}
                      />
                      <div className="space-y-2 text-xs text-foreground/60">
                        {preorder.references?.length ? (
                          preorder.references.map((ref) => (
                            <a
                              key={ref.id}
                              href={ref.file || "#"}
                              className="block text-primary"
                              target="_blank"
                              rel="noreferrer"
                            >
                              {ref.original_filename || "Reference file"}
                            </a>
                          ))
                        ) : (
                          <p>No reference files uploaded yet.</p>
                        )}
                      </div>
                    </div>
                  </div>
                </Card>

                <Card variant="bordered" className="space-y-4">
                  <h2 className="text-lg font-semibold">Messages</h2>
                  <div className="space-y-3">
                    {preorder.messages?.length ? (
                      preorder.messages.map((message) => (
                        <div
                          key={message.id}
                          className="rounded-xl border border-border bg-muted/30 p-3 text-sm"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-foreground/60">
                            <span>{message.sender_name || "Team"}</span>
                            <span>{formatDateTime(message.created_at)}</span>
                          </div>
                          {message.subject ? (
                            <p className="mt-2 font-semibold">{message.subject}</p>
                          ) : null}
                          <p className="mt-1 text-foreground/70">{message.message}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-foreground/60">
                        No messages yet. Start the conversation below.
                      </p>
                    )}
                  </div>
                  <div className="grid gap-3">
                    <input
                      className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      placeholder="Subject (optional)"
                      value={messageSubject}
                      onChange={(event) => setMessageSubject(event.target.value)}
                    />
                    <textarea
                      className="min-h-[120px] w-full rounded-lg border border-border bg-card px-3 py-2 text-sm"
                      placeholder="Write your message"
                      value={messageBody}
                      onChange={(event) => setMessageBody(event.target.value)}
                    />
                    <div>
                      <Button
                        variant="primary-gradient"
                        onClick={handleSendMessage}
                        disabled={actions.sendMessage.isPending}
                      >
                        {actions.sendMessage.isPending ? "Sending..." : "Send message"}
                      </Button>
                    </div>
                  </div>
                </Card>
              </div>

              <div className="space-y-6">
                <Card variant="bordered" className="space-y-3">
                  <h2 className="text-lg font-semibold">Payment snapshot</h2>
                  <div className="space-y-2 text-sm text-foreground/70">
                    <p>
                      Estimated total: {formatMoney(preorder.estimated_price || 0, currency)}
                    </p>
                    {preorder.final_price ? (
                      <p>Final quote: {formatMoney(preorder.final_price, currency)}</p>
                    ) : null}
                    <p>
                      Deposit required: {formatMoney(preorder.deposit_required || 0, currency)}
                    </p>
                    <p>
                      Paid to date: {formatMoney(preorder.amount_paid || 0, currency)}
                    </p>
                    <p className="font-semibold">
                      Remaining: {formatMoney(preorder.amount_remaining || 0, currency)}
                    </p>
                  </div>
                </Card>

                {statusHistory.length ? (
                  <Card variant="bordered" className="space-y-3">
                    <h2 className="text-lg font-semibold">Status timeline</h2>
                    <div className="space-y-3 text-sm text-foreground/70">
                      {statusHistory.map((entry) => (
                        <div key={entry.id} className="border-l-2 border-primary/40 pl-3">
                          <p className="text-xs uppercase tracking-[0.2em] text-foreground/60">
                            {formatDateTime(entry.created_at)}
                          </p>
                          <p className="font-medium">
                            {entry.to_status_display || entry.to_status}
                          </p>
                          {entry.notes ? (
                            <p className="text-xs text-foreground/60">{entry.notes}</p>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </Card>
                ) : null}

                <Card variant="bordered" className="space-y-3">
                  <h2 className="text-lg font-semibold">Shipping</h2>
                  <div className="text-sm text-foreground/70">
                    <p>{preorder.shipping_method || "Standard"}</p>
                    <p className="mt-2">
                      {joinAddress(
                        preorder.shipping_address_line_1,
                        preorder.shipping_address_line_2,
                        preorder.shipping_city,
                        preorder.shipping_state,
                        preorder.shipping_postal_code,
                        preorder.shipping_country
                      ) || "Shipping details not provided."}
                    </p>
                    {preorder.tracking_number ? (
                      <p className="mt-2">Tracking: {preorder.tracking_number}</p>
                    ) : null}
                    {preorder.tracking_url ? (
                      <a
                        href={preorder.tracking_url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-1 inline-block text-primary"
                      >
                        Track shipment
                      </a>
                    ) : null}
                  </div>
                </Card>

                {preorder.quotes?.length ? (
                  <Card variant="bordered" className="space-y-4">
                    <h2 className="text-lg font-semibold">Quotes</h2>
                    <div className="space-y-4">
                      {preorder.quotes.map((quote) => (
                        <div key={quote.id} className="rounded-xl border border-border p-3 text-sm">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="font-semibold">{quote.quote_number}</span>
                            <span className="rounded-full bg-muted px-2 py-1 text-xs">
                              {quote.status}
                            </span>
                          </div>
                          <div className="mt-2 space-y-1 text-foreground/70">
                            <p>Total: {formatMoney(quote.total || 0, currency)}</p>
                            <p>Valid until: {formatDate(quote.valid_until)}</p>
                          </div>
                          {quote.terms ? (
                            <p className="mt-2 text-xs text-foreground/60">{quote.terms}</p>
                          ) : null}
                          {quote.status === "pending" || quote.status === "sent" ? (
                            <div className="mt-3 space-y-2">
                              <textarea
                                className="min-h-[80px] w-full rounded-lg border border-border bg-card px-3 py-2 text-xs"
                                placeholder="Optional rejection reason"
                                value={rejectReasons[quote.id] || ""}
                                onChange={(event) =>
                                  setRejectReasons((prev) => ({
                                    ...prev,
                                    [quote.id]: event.target.value,
                                  }))
                                }
                              />
                              <div className="flex flex-wrap gap-2">
                                <Button
                                  size="sm"
                                  variant="primary-gradient"
                                  onClick={() => handleAcceptQuote(quote)}
                                  disabled={actions.acceptQuote.isPending}
                                >
                                  Accept quote
                                </Button>
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => handleRejectQuote(quote)}
                                  disabled={actions.rejectQuote.isPending}
                                >
                                  Reject quote
                                </Button>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </Card>
                ) : null}

                {preorder.payments?.length ? (
                  <Card variant="bordered" className="space-y-3">
                    <h2 className="text-lg font-semibold">Payments</h2>
                    <div className="space-y-2 text-sm text-foreground/70">
                      {preorder.payments.map((payment) => (
                        <div key={payment.id} className="flex items-center justify-between">
                          <div>
                            <p className="font-medium">{payment.payment_type}</p>
                            <p className="text-xs text-foreground/60">
                              {formatDateTime(payment.created_at)}
                            </p>
                          </div>
                          <div className="text-right">
                            <p>{formatMoney(payment.amount || 0, currency)}</p>
                            <p className="text-xs text-foreground/60">
                              {payment.status}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                ) : null}

                <Card variant="bordered" className="space-y-3">
                  <h2 className="text-lg font-semibold">Need help?</h2>
                  <p className="text-sm text-foreground/70">
                    Track your preorder publicly or start a new request.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Button asChild size="sm" variant="secondary">
                      <Link href="/preorders/track/">Track preorder</Link>
                    </Button>
                    <Button asChild size="sm" variant="secondary">
                      <Link href="/preorders/create/1/">Start another preorder</Link>
                    </Button>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        ) : (
          <Card variant="bordered" className="p-6 text-sm text-foreground/70">
            Preorder not found.
          </Card>
        )}
      </div>
    </AuthGate>
  );
}
