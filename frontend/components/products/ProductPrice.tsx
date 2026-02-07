import { cn } from "@/lib/utils";
import { formatMoney } from "@/lib/money";

export function ProductPrice({
  price,
  salePrice,
  currentPrice,
  currency,
  className,
  priceClassName,
}: {
  price?: string | null;
  salePrice?: string | null;
  currentPrice?: string | null;
  currency: string;
  className?: string;
  priceClassName?: string;
}) {
  const base = currentPrice || salePrice || price || "";
  const showSale = Boolean(salePrice && price && salePrice !== price);

  return (
    <div className={cn("flex flex-wrap items-baseline gap-2", className)}>
      <span className={cn("text-lg font-semibold", priceClassName)}>
        {formatMoney(base, currency)}
      </span>
      {showSale ? (
        <span className="text-sm text-foreground/50 line-through">
          {formatMoney(price || "", currency)}
        </span>
      ) : null}
    </div>
  );
}
