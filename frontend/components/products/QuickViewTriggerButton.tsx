"use client";

import * as React from "react";
import { Button } from "@/components/ui/Button";
import { QuickViewModal } from "@/components/products/QuickViewModal";

export function QuickViewTriggerButton({
  slug,
  className,
  label = "View",
}: {
  slug: string;
  className?: string;
  label?: string;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <Button
        size="sm"
        variant="secondary"
        className={className}
        onClick={() => setOpen(true)}
      >
        {label}
      </Button>
      <QuickViewModal
        slug={open ? slug : null}
        isOpen={open}
        onClose={() => setOpen(false)}
      />
    </>
  );
}

