"""SERP analysis, feature detection, intent classification and brief generation."""
import re
import time
import requests
from collections import Counter
from django.utils import timezone
from .models import Keyword, SERPSnapshot, ContentBrief

# Minimal stopword set for English (extend as needed)
STOPWORDS = set("""
the a an and or in on for with from by to of at is it this that these those how why what when where who which
""".split())

QUESTION_WORDS = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
TRANSACTIONAL_HINTS = {'buy', 'price', 'purchase', 'discount', 'coupon', 'deal', 'order', 'shipping', 'sale'}


def detect_serp_features(keyword_term, date=None):
    """Analyze recent SERP snapshots for presence of features.

    Returns a dict with detected features like featured_snippet, people_also_ask, knowledge_panel, shopping.
    """
    q = Keyword.objects.filter(term=keyword_term).first()
    if not q:
        return {}
    if date is None:
        date = timezone.now().date()

    rows = SERPSnapshot.objects.filter(keyword=q, date=date)
    features = {
        'featured_snippet': False,
        'people_also_ask': False,
        'knowledge_panel': False,
        'shopping': False,
        'image_pack': False,
    }

    for r in rows:
        raw = r.raw or {}
        # SerpAPI keys
        if isinstance(raw, dict):
            if raw.get('featured_snippet') or raw.get('is_answer_box'):
                features['featured_snippet'] = True
            # serpapi may include 'related_questions'
            if raw.get('related_questions'):
                features['people_also_ask'] = True
            if raw.get('knowledge_graph'):
                features['knowledge_panel'] = True
            if raw.get('shopping_results'):
                features['shopping'] = True
            # Detect image pack or local pack from serpapi
            if raw.get('image_results'):
                features['image_pack'] = True
        # Heuristic checks from snippet/url
        snippet = (r.snippet or '').lower()
        url = (r.url or '').lower()
        if '?' in snippet and len(snippet.split()) < 40:
            features['people_also_ask'] = True
        if any('/product/' in url or '/products/' in url or '/p/' in url or '/shop/' in url for url in [url]):
            features['shopping'] = True
    return features


def classify_intent_from_term_and_serp(keyword_term, date=None):
    """Heuristic intent classifier using the term and SERP signals."""
    term = (keyword_term or '').lower()
    if any(w in term for w in TRANSACTIONAL_HINTS) or term.startswith('best ') or 'review' in term:
        return 'transactional'
    if 'near me' in term or re.search(r'\b\d{5}\b', term):
        return 'navigational'
    if any(term.strip().startswith(qw + ' ') or term.strip().endswith('?') for qw in QUESTION_WORDS):
        return 'informational'

    # fallback to SERP presence: if many product pages in top results -> transactional
    q = Keyword.objects.filter(term=keyword_term).first()
    if not q:
        return 'informational'

    if date is None:
        date = timezone.now().date()
    rows = SERPSnapshot.objects.filter(keyword=q, date=date)[:8]
    product_like = 0
    question_like = 0
    for r in rows:
        url = (r.url or '').lower()
        snippet = (r.snippet or '').lower()
        if any(p in url for p in ['/product', '/products', '/shop', '/buy']) or any(p in snippet for p in TRANSACTIONAL_HINTS):
            product_like += 1
        if '?' in snippet or any(qw in snippet for qw in QUESTION_WORDS):
            question_like += 1
    if product_like >= 3:
        return 'transactional'
    if question_like >= 2:
        return 'informational'
    return 'informational'


def _simple_tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text.lower())
    tokens = [t for t in text.split() if t and t not in STOPWORDS and not t.isdigit() and len(t) > 1]
    return tokens


def generate_content_brief(keyword_term, date=None, top_n=5):
    """Generate a content brief for the keyword using top N result pages.

    Saves ContentBrief and returns the instance.
    """
    q = Keyword.objects.filter(term=keyword_term).first()
    if not q:
        q = Keyword.objects.create(term=keyword_term)

    if date is None:
        date = timezone.now().date()

    rows = SERPSnapshot.objects.filter(keyword=q, date=date).order_by('position')[:top_n]
    urls = []
    headings_counter = Counter()
    terms_counter = Counter()
    word_counts = []

    for r in rows:
        url = r.url
        if not url:
            continue
        urls.append(url)
        # Fetch page
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; BunoraaSEO/1.0)'}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            html = resp.text
            # BeautifulSoup lazily
            try:
                from bs4 import BeautifulSoup
            except Exception:
                raise RuntimeError('beautifulsoup4 is required for content brief generation (pip install beautifulsoup4)')
            soup = BeautifulSoup(html, 'html.parser')
            # Extract headings
            for h in soup.find_all(['h1', 'h2', 'h3']):
                text = (h.get_text(strip=True) or '')[:200]
                if text:
                    headings_counter[text] += 1
            # Extract main text
            paragraphs = soup.find_all('p')
            page_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
            tokens = _simple_tokenize(page_text)
            terms_counter.update(tokens)
            word_counts.append(len(tokens))
            # Sleep briefly to avoid hammering
            time.sleep(0.5)
        except Exception as exc:
            # skip problematic pages
            continue

    # Build suggested headings from the most common headings
    suggested_headings = [h for h, _ in headings_counter.most_common(10)]
    top_terms = [t for t, _ in terms_counter.most_common(30)]
    rec_wc = int(sum(word_counts) / len(word_counts)) if word_counts else None

    brief = ContentBrief.objects.create(
        keyword=q,
        generated_by='analysis.generate_content_brief',
        top_urls=urls,
        suggested_headings=suggested_headings,
        top_terms=top_terms,
        recommended_word_count=rec_wc,
        notes='Brief generated from top SERP results.'
    )

    return brief