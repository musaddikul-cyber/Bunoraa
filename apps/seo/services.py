import os
import time
import re
import requests
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from django.conf import settings
from django.utils import timezone
from .models import Keyword, SERPSnapshot, ContentBrief


USER_AGENT = os.environ.get('SEO_USER_AGENT', 'Mozilla/5.0 (compatible; BunoraaSEO/1.0; +https://bunoraa.com)')
DEFAULT_PRERENDER_USER_AGENT = "Mozilla/5.0 (compatible; BunoraaPrerender/1.0)"
STOPWORDS = set("""
the a an and or in on for with from by to of at is it this that these those how why what when where who which
""".split())
QUESTION_WORDS = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
TRANSACTIONAL_HINTS = {'buy', 'price', 'purchase', 'discount', 'coupon', 'deal', 'order', 'shipping', 'sale'}


def fetch_serp_google(query, num=10, country=None):
    """Basic SERP fetcher via requests + parser (lightweight). Use SERPAPI if API key set."""
    from serpapi import GoogleSearch

    serp_api_key = os.environ.get('SERPAPI_KEY')
    results = []

    if serp_api_key:
        # Use SerpAPI for reliable results
        params = {"engine": "google", "q": query, "num": num, "api_key": serp_api_key}
        if country:
            params['gl'] = country
        search = GoogleSearch(params)
        data = search.get_dict()
        organic = data.get('organic_results', [])
        for idx, r in enumerate(organic[:num], start=1):
            results.append({'position': idx, 'title': r.get('title'), 'url': r.get('link'), 'snippet': r.get('snippet'), 'raw': r})
        return results

    # Fallback: simple scraping (best-effort). Note: fragile and for low-volume use only
    q = requests.utils.requote_uri(query)
    url = f'https://www.google.com/search?q={q}&num={num}'
    headers = {'User-Agent': USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Google markup changes often; try popular selectors
    divs = soup.select('div[data-attrid]') or soup.select('div.g')
    rank = 0
    for g in soup.select('div.g'):
        a = g.select_one('a')
        if not a or not a.get('href'):
            continue
        rank += 1
        title = g.select_one('h3')
        title_text = title.get_text(strip=True) if title else ''
        snippet_el = g.select_one('.IsZvec') or g.select_one('.VwiC3b')
        snippet = snippet_el.get_text(separator=' ', strip=True) if snippet_el else ''
        results.append({'position': rank, 'title': title_text, 'url': a.get('href'), 'snippet': snippet, 'raw': None})
        if rank >= num:
            break
    time.sleep(1)
    return results


def snapshot_keyword_serp(keyword_term, num=10):
    k, _ = Keyword.objects.get_or_create(term=keyword_term)
    rows = fetch_serp_google(keyword_term, num=num)
    date = timezone.now().date()
    saved = []
    for r in rows:
        obj = SERPSnapshot.objects.create(
            keyword=k, date=date, position=r['position'], url=r['url'], title=r.get('title')[:512], snippet=r.get('snippet', '')[:2000], raw=r.get('raw'), source=('serpapi' if os.environ.get('SERPAPI_KEY') else 'scrape')
        )
        saved.append(obj)
    return saved


def detect_serp_features(keyword_term, date=None):
    """Analyze recent SERP snapshots for presence of rich features."""
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
        if isinstance(raw, dict):
            if raw.get('featured_snippet') or raw.get('is_answer_box'):
                features['featured_snippet'] = True
            if raw.get('related_questions'):
                features['people_also_ask'] = True
            if raw.get('knowledge_graph'):
                features['knowledge_panel'] = True
            if raw.get('shopping_results'):
                features['shopping'] = True
            if raw.get('image_results'):
                features['image_pack'] = True

        snippet = (r.snippet or '').lower()
        url = (r.url or '').lower()
        if '?' in snippet and len(snippet.split()) < 40:
            features['people_also_ask'] = True
        if any(marker in url for marker in ('/product/', '/products/', '/p/', '/shop/')):
            features['shopping'] = True
    return features


def classify_intent_from_term_and_serp(keyword_term, date=None):
    """Heuristic intent classifier using term text and SERP composition."""
    term = (keyword_term or '').lower()
    if any(w in term for w in TRANSACTIONAL_HINTS) or term.startswith('best ') or 'review' in term:
        return 'transactional'
    if 'near me' in term or re.search(r'\b\d{5}\b', term):
        return 'navigational'
    if any(term.strip().startswith(qw + ' ') or term.strip().endswith('?') for qw in QUESTION_WORDS):
        return 'informational'

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
        if any(marker in url for marker in ('/product', '/products', '/shop', '/buy')) or any(
            marker in snippet for marker in TRANSACTIONAL_HINTS
        ):
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
    return [t for t in text.split() if t and t not in STOPWORDS and not t.isdigit() and len(t) > 1]


def generate_content_brief(keyword_term, date=None, top_n=5):
    """Generate and store a content brief based on top SERP pages."""
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
        try:
            headers = {'User-Agent': USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            for h in soup.find_all(['h1', 'h2', 'h3']):
                text = (h.get_text(strip=True) or '')[:200]
                if text:
                    headings_counter[text] += 1
            paragraphs = soup.find_all('p')
            page_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
            tokens = _simple_tokenize(page_text)
            terms_counter.update(tokens)
            word_counts.append(len(tokens))
            time.sleep(0.5)
        except Exception:
            continue

    suggested_headings = [h for h, _ in headings_counter.most_common(10)]
    top_terms = [t for t, _ in terms_counter.most_common(30)]
    rec_wc = int(sum(word_counts) / len(word_counts)) if word_counts else None

    return ContentBrief.objects.create(
        keyword=q,
        generated_by='services.generate_content_brief',
        top_urls=urls,
        suggested_headings=suggested_headings,
        top_terms=top_terms,
        recommended_word_count=rec_wc,
        notes='Brief generated from top SERP results.',
    )


def is_prerender_enabled() -> bool:
    return bool(getattr(settings, "PRERENDER_ENABLED", False))


def get_cache_dir() -> Path:
    cache_dir = Path(settings.BASE_DIR) / getattr(settings, "PRERENDER_CACHE_DIR", "prerender_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def normalize_path(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def path_to_filename(path: str) -> str:
    key = path.strip("/").replace("/", "_") or "index"
    return f"{key}.html"


def prerender_paths(
    *,
    paths: list[str],
    timeout: int = 15,
    user_agent: str = DEFAULT_PRERENDER_USER_AGENT,
) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
    site_url = getattr(settings, "SITE_URL", "https://bunoraa.com")
    cache_dir = get_cache_dir()
    headers = {"User-Agent": user_agent}

    saved = 0
    successes = []
    failures = []

    for raw_path in paths:
        path = normalize_path(raw_path)
        url = urljoin(site_url, path.lstrip("/"))
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            output_path = cache_dir / path_to_filename(path)
            output_path.write_bytes(response.content)
            successes.append((path, str(output_path)))
            saved += 1
        except Exception as exc:
            failures.append((url, str(exc)))

    return saved, successes, failures
