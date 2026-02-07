import Link from "next/link";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex w-full max-w-3xl flex-col items-center gap-4 px-6 py-20 text-center">
        <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
          404
        </p>
        <h1 className="text-3xl font-semibold">Page not found</h1>
        <p className="text-sm text-foreground/70">
          The page you are looking for does not exist.
        </p>
        <Button asChild variant="primary-gradient">
          <Link href="/">Go home</Link>
        </Button>
      </div>
    </div>
  );
}
