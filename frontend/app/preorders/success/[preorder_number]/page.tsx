import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export default async function PreorderSuccessPage({
  params,
}: {
  params: Promise<{ preorder_number: string }>;
}) {
  const { preorder_number } = await params;
  return (
    <div className="mx-auto w-full max-w-lg px-4 py-12 sm:px-6 sm:py-20">
      <Card variant="bordered" className="space-y-4 p-5 text-center sm:p-6">
        <p className="text-xs uppercase tracking-[0.3em] text-foreground/60">
          Preorder submitted
        </p>
        <h1 className="text-xl font-semibold sm:text-2xl">
          We received your request
        </h1>
        <p className="text-sm text-foreground/70">
          Your preorder number is <strong>{preorder_number}</strong>.
        </p>
        <p className="text-sm text-foreground/60">
          Our team will review the details and send a quote and timeline to your
          email. No payment is collected until the quote is approved.
        </p>
        <div className="flex flex-col gap-3">
          <Button asChild variant="primary-gradient">
            <Link href={`/preorders/track/?order=${preorder_number}`}>Track preorder</Link>
          </Button>
          <Button asChild variant="secondary">
            <Link href="/preorders/my-orders/">View my preorders</Link>
          </Button>
          <Button asChild variant="ghost">
            <Link href="/preorders/">Back to preorders</Link>
          </Button>
        </div>
      </Card>
    </div>
  );
}
