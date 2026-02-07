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
    <div className="mb-6">
      <div className="flex items-center justify-start gap-4 overflow-x-auto pb-2 scrollbar-hide md:justify-center">
        {steps.map((step, index) => {
          const isComplete = index < currentIndex;
          const isActive = index === currentIndex;
          const isClickable = Boolean(onStepClick && index <= currentIndex);
          return (
            <div key={step.key} className="flex items-center">
              <div
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-full text-xs font-semibold sm:h-10 sm:w-10 sm:text-sm",
                  isComplete
                    ? "bg-emerald-600 text-white"
                    : isActive
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground/50",
                  isClickable && "cursor-pointer"
                )}
                onClick={() => {
                  if (isClickable) onStepClick?.(step.key);
                }}
                role={isClickable ? "button" : undefined}
                tabIndex={isClickable ? 0 : undefined}
                onKeyDown={(event) => {
                  if (!isClickable) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onStepClick?.(step.key);
                  }
                }}
              >
                {isComplete ? (
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
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
              </div>
              <span
                className={cn(
                  "ml-2 text-xs font-medium sm:text-sm",
                  isActive ? "text-foreground" : "text-foreground/50",
                  isClickable && "cursor-pointer hover:text-foreground"
                )}
                onClick={() => {
                  if (isClickable) onStepClick?.(step.key);
                }}
                role={isClickable ? "button" : undefined}
                tabIndex={isClickable ? 0 : undefined}
                onKeyDown={(event) => {
                  if (!isClickable) return;
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onStepClick?.(step.key);
                  }
                }}
              >
                {step.label}
              </span>
              {index < steps.length - 1 ? (
                <div className="mx-3 h-0.5 w-6 bg-border sm:w-10 md:w-16" />
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
