import { Card } from "@/components/ui/Card";

export default async function SharedCartPage({
  params,
}: {
  params: Promise<{ token: string }>;
}) {
  const { token } = await params;
  return (
    <div className="mx-auto w-full max-w-3xl px-4 sm:px-6 py-12">
      <Card variant="bordered" className="p-6 text-sm text-foreground/70">
        Shared carts require a dedicated API endpoint. Token: {token}
      </Card>
    </div>
  );
}
