from django.core.management.base import BaseCommand, CommandError
from apps.seo.analysis import generate_content_brief, detect_serp_features, classify_intent_from_term_and_serp


class Command(BaseCommand):
    help = 'Generate content brief for a keyword based on current SERP and store it in DB'

    def add_arguments(self, parser):
        parser.add_argument('keyword', type=str, help='Keyword to generate brief for')
        parser.add_argument('--top', type=int, default=5, help='Top N pages to use')

    def handle(self, *args, **options):
        kw = options['keyword']
        top = options['top']
        try:
            brief = generate_content_brief(kw, top_n=top)
            features = detect_serp_features(kw)
            intent = classify_intent_from_term_and_serp(kw)
            self.stdout.write(self.style.SUCCESS(f'Generated brief id={brief.id} for "{kw}"; intent={intent}; features={features}'))
        except Exception as exc:
            raise CommandError(f'Error generating brief: {exc}')