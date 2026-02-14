import { apiFetch } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { JsonLd } from "@/components/seo/JsonLd";
import { cleanObject } from "@/lib/seo";

export const revalidate = 300;

type Faq = {
  id: string;
  question: string;
  answer: string;
  category?: string | null;
};

async function getFaqs() {
  const response = await apiFetch<Faq[]>("/pages/faqs/", {
    next: { revalidate },
  });
  return response.data;
}

export default async function FaqPage() {
  const faqs = await getFaqs();
  const faqSchema = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((faq) =>
      cleanObject({
        "@type": "Question",
        name: faq.question,
        acceptedAnswer: cleanObject({
          "@type": "Answer",
          text: faq.answer,
        }),
      })
    ),
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-12">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.2em] text-foreground/60">
            FAQ
          </p>
          <h1 className="text-3xl font-semibold">Frequently asked questions</h1>
        </div>

        <div className="space-y-4">
          {faqs.map((faq) => (
            <Card key={faq.id} variant="bordered" className="space-y-2">
              <h2 className="text-lg font-semibold">{faq.question}</h2>
              <p className="text-sm text-foreground/70">{faq.answer}</p>
            </Card>
          ))}
        </div>
      </div>
      {faqs.length ? <JsonLd data={faqSchema} /> : null}
    </div>
  );
}
