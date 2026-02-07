import { Card } from "@/components/ui/Card";

export function ProductCardSkeleton() {
  return (
    <Card variant="bordered" className="flex flex-col gap-4">
      <div className="aspect-[4/5] rounded-xl bg-muted skeleton" />
      <div className="h-4 w-24 rounded bg-muted skeleton" />
      <div className="h-5 w-3/4 rounded bg-muted skeleton" />
      <div className="h-4 w-1/2 rounded bg-muted skeleton" />
      <div className="h-9 w-32 rounded bg-muted skeleton" />
    </Card>
  );
}