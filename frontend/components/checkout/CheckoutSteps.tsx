import { cn } from "@/lib/utils";

type Step = "information" | "shipping" | "payment" | "review";

const steps: { key: Step; label: string; number: number }[] = [
  { key: "information", label: "Information", number: 1 },
  { key: "shipping", label: "Shipping", number: 2 },
  { key: "payment", label: "Payment", number: 3 },
  { key: "review", label: "Review", number: 4 },
];

export function CheckoutSteps({
  current,
  onStepClick,
}: {
  current: Step;
  onStepClick?: (step: Step) => void;
}) {
  const currentIndex = steps.findIndex((step) => step.key === current);

  return (
    <div className="mb-1 lg:mb-6">
      <ol className="flex items-center justify-start gap-2 overflow-x-auto pb-1 scrollbar-hide md:justify-center">
        {steps.map((step, index) => {
          const isComplete = index < currentIndex;
          const isActive = index === currentIndex;
          const isClickable = Boolean(onStepClick && index <= currentIndex);
          return (
            <li key={step.key} className="shrink-0">
              <button
                type="button"
                className={cn(
                  "inline-flex min-h-11 items-center gap-2 rounded-full border px-3 py-2 text-xs font-medium sm:min-h-10 sm:text-sm",
                  isComplete
                    ? "border-emerald-600 bg-emerald-600 text-white"
                    : isActive
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-muted text-foreground/70",
                  isClickable && "cursor-pointer",
                  !isClickable && "cursor-default"
                )}
                disabled={!isClickable}
                aria-current={isActive ? "step" : undefined}
                aria-label={`Step ${step.number}: ${step.label}`}
                onClick={() => {
                  if (isClickable) onStepClick?.(step.key);
                }}
                onKeyDown={(event) => {
                  if (!isClickable) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onStepClick?.(step.key);
                  }
                }}
              >
                <span
                  className={cn(
                    "inline-flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold",
                    isComplete || isActive
                      ? "bg-white/20 text-current"
                      : "bg-background text-foreground/70"
                  )}
                >
                  {isComplete ? (
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  ) : (
                    step.number
                  )}
                </span>
                <span className="whitespace-nowrap">{step.label}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
