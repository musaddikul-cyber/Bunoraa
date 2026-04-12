import os
import time
import re
import json
import hashlib
import logging
import tempfile
import threading
import requests
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
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


LOGGER = logging.getLogger("bunoraa.seo.prerender")
MANIFEST_FILENAME = "manifest.json"
SNAPSHOT_DIRNAME = "snapshots"
_REFRESH_GUARD = threading.Lock()
_REFRESH_INFLIGHT: set[str] = set()


@dataclass(frozen=True)
class PrerenderConfig:
    site_url: str
    timeout_seconds: int
    connect_timeout_seconds: int
    read_timeout_seconds: int
    retries: int
    backoff_seconds: float
    user_agent: str
    fresh_ttl_seconds: int
    stale_while_revalidate_seconds: int
    stale_if_error_seconds: int
    max_content_bytes: int
    allowed_query_keys: tuple[str, ...]
    on_demand_enabled: bool


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_prerender_enabled() -> bool:
    return bool(getattr(settings, "PRERENDER_ENABLED", False))


def get_prerender_config(
    *,
    timeout_override: int | None = None,
    user_agent_override: str | None = None,
    retries_override: int | None = None,
) -> PrerenderConfig:
    site_url = str(getattr(settings, "SITE_URL", "https://bunoraa.com")).strip() or "https://bunoraa.com"
    timeout_seconds = _safe_int(
        timeout_override if timeout_override is not None else getattr(settings, "PRERENDER_TIMEOUT_SECONDS", 15),
        15,
    )
    retries = _safe_int(
        retries_override if retries_override is not None else getattr(settings, "PRERENDER_FETCH_RETRIES", 2),
        2,
    )
    allowed_query_keys = tuple(
        str(item).strip()
        for item in getattr(settings, "PRERENDER_ALLOWED_QUERY_KEYS", ())
        if str(item).strip()
    )
    return PrerenderConfig(
        site_url=site_url,
        timeout_seconds=max(timeout_seconds, 1),
        connect_timeout_seconds=max(_safe_int(getattr(settings, "PRERENDER_CONNECT_TIMEOUT_SECONDS", 5), 5), 1),
        read_timeout_seconds=max(_safe_int(getattr(settings, "PRERENDER_READ_TIMEOUT_SECONDS", timeout_seconds), timeout_seconds), 1),
        retries=max(retries, 0),
        backoff_seconds=max(_safe_float(getattr(settings, "PRERENDER_FETCH_BACKOFF_SECONDS", 0.5), 0.5), 0.0),
        user_agent=(user_agent_override or getattr(settings, "PRERENDER_USER_AGENT", DEFAULT_PRERENDER_USER_AGENT)),
        fresh_ttl_seconds=max(_safe_int(getattr(settings, "PRERENDER_FRESH_TTL_SECONDS", 3600), 3600), 0),
        stale_while_revalidate_seconds=max(
            _safe_int(getattr(settings, "PRERENDER_STALE_WHILE_REVALIDATE_SECONDS", 3600), 3600),
            0,
        ),
        stale_if_error_seconds=max(_safe_int(getattr(settings, "PRERENDER_STALE_IF_ERROR_SECONDS", 86400), 86400), 0),
        max_content_bytes=max(_safe_int(getattr(settings, "PRERENDER_MAX_CONTENT_BYTES", 4_194_304), 4_194_304), 1024),
        allowed_query_keys=allowed_query_keys,
        on_demand_enabled=bool(getattr(settings, "PRERENDER_ON_DEMAND_ENABLED", True)),
    )


def get_cache_dir() -> Path:
    configured = Path(str(getattr(settings, "PRERENDER_CACHE_DIR", "prerender_cache")))
    cache_dir = configured if configured.is_absolute() else Path(settings.BASE_DIR) / configured
    (cache_dir / SNAPSHOT_DIRNAME).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _snapshots_dir() -> Path:
    return get_cache_dir() / SNAPSHOT_DIRNAME


def _manifest_path() -> Path:
    return get_cache_dir() / MANIFEST_FILENAME


def normalize_path(
    path: str,
    *,
    allow_query: bool = True,
    allowed_query_keys: tuple[str, ...] | None = None,
) -> str:
    candidate = str(path or "").strip()
    if not candidate:
        return "/"

    parsed = urlsplit(candidate)
    normalized_path = parsed.path or "/"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    normalized_path = re.sub(r"/{2,}", "/", normalized_path)

    # Keep trailing slash for route paths; preserve file-like paths.
    if normalized_path != "/" and not normalized_path.endswith("/"):
        leaf = normalized_path.rsplit("/", 1)[-1]
        if "." not in leaf:
            normalized_path = f"{normalized_path}/"

    if not allow_query:
        return normalized_path

    raw_query = parsed.query or ""
    effective_allowed = tuple(item.strip() for item in (allowed_query_keys or ()) if item and item.strip())
    if not raw_query or not effective_allowed:
        return normalized_path

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(raw_query, keep_blank_values=True)
        if key in effective_allowed
    ]
    if not query_pairs:
        return normalized_path

    query_pairs.sort()
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit(("", "", normalized_path, query, ""))


def path_to_cache_key(path: str) -> str:
    config = get_prerender_config()
    normalized = normalize_path(path, allow_query=True, allowed_query_keys=config.allowed_query_keys)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def path_to_filename(path: str) -> str:
    return f"{path_to_cache_key(path)}.html"


def _metadata_filename(path: str) -> str:
    return f"{path_to_cache_key(path)}.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(target.parent), prefix=f"{target.name}.tmp-", delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, target)


def _atomic_write_text(target: Path, payload: str) -> None:
    _atomic_write_bytes(target, payload.encode("utf-8"))


def _update_manifest_entry(path: str, metadata: dict) -> None:
    manifest_file = _manifest_path()
    manifest = _read_json(manifest_file)
    entries = manifest if isinstance(manifest, dict) else {}
    entries[path] = {
        "path": path,
        "key": metadata.get("key"),
        "updated_at": metadata.get("generated_at"),
        "expires_at": metadata.get("expires_at"),
        "stale_if_error_until": metadata.get("stale_if_error_until"),
        "etag": metadata.get("etag"),
        "url": metadata.get("url"),
        "html_file": metadata.get("html_file"),
        "metadata_file": metadata.get("metadata_file"),
    }
    _atomic_write_text(manifest_file, json.dumps(entries, indent=2, sort_keys=True))


def list_prerender_entries() -> list[dict]:
    manifest = _read_json(_manifest_path())
    if not isinstance(manifest, dict):
        return []
    entries = list(manifest.values())
    entries.sort(key=lambda item: str(item.get("path", "")))
    return entries


def build_prerender_url(path: str, *, site_url: str | None = None) -> str:
    config = get_prerender_config()
    origin = (site_url or config.site_url).rstrip("/") + "/"
    normalized = normalize_path(path, allow_query=True, allowed_query_keys=config.allowed_query_keys)
    return urljoin(origin, normalized.lstrip("/"))


def _snapshot_paths(path: str) -> tuple[str, Path, Path]:
    normalized = normalize_path(path, allow_query=True, allowed_query_keys=get_prerender_config().allowed_query_keys)
    key = path_to_cache_key(normalized)
    snapshots_dir = _snapshots_dir()
    html_path = snapshots_dir / path_to_filename(normalized)
    metadata_path = snapshots_dir / _metadata_filename(normalized)
    return normalized, html_path, metadata_path


def load_prerender_snapshot(path: str) -> dict | None:
    normalized, html_path, metadata_path = _snapshot_paths(path)
    if not metadata_path.exists() or not html_path.exists():
        return None
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        return None
    try:
        content = html_path.read_bytes()
    except Exception:
        return None

    if not metadata.get("etag"):
        metadata["etag"] = f"\"{hashlib.sha256(content).hexdigest()}\""
    metadata.setdefault("path", normalized)
    metadata.setdefault("generated_at", 0)
    metadata.setdefault("expires_at", 0)
    metadata.setdefault("stale_while_revalidate_until", metadata.get("expires_at", 0))
    metadata.setdefault("stale_if_error_until", metadata.get("stale_while_revalidate_until", 0))
    metadata.setdefault("status_code", 200)
    metadata.setdefault("headers", {})
    metadata["content"] = content
    metadata["html_path"] = str(html_path)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def get_snapshot_state(snapshot: dict, now_ts: int | None = None) -> str:
    now = now_ts if now_ts is not None else int(time.time())
    expires_at = _safe_int(snapshot.get("expires_at"), 0)
    swr_until = _safe_int(snapshot.get("stale_while_revalidate_until"), 0)
    sie_until = _safe_int(snapshot.get("stale_if_error_until"), 0)
    if now <= expires_at:
        return "fresh"
    if now <= swr_until:
        return "stale-while-revalidate"
    if now <= sie_until:
        return "stale-if-error"
    return "expired"


def _parse_cache_control(header_value: str) -> dict[str, int]:
    directives: dict[str, int] = {}
    if not header_value:
        return directives
    for token in header_value.split(","):
        part = token.strip().lower()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        directives[key.strip()] = _safe_int(value.strip().strip('"'), 0)
    return directives


def _derive_ttls(config: PrerenderConfig, source_cache_control: str) -> tuple[int, int, int]:
    parsed = _parse_cache_control(source_cache_control)
    fresh = parsed.get("s-maxage", parsed.get("max-age", config.fresh_ttl_seconds))
    swr = parsed.get("stale-while-revalidate", config.stale_while_revalidate_seconds)
    sie = parsed.get("stale-if-error", config.stale_if_error_seconds)
    max_ttl = max(_safe_int(getattr(settings, "PRERENDER_MAX_TTL_SECONDS", 604800), 604800), 60)
    return max(min(fresh, max_ttl), 0), max(min(swr, max_ttl), 0), max(min(sie, max_ttl), 0)


def _save_snapshot(path: str, url: str, response: requests.Response, content: bytes, config: PrerenderConfig) -> dict:
    normalized, html_path, metadata_path = _snapshot_paths(path)
    now = int(time.time())
    cache_control = response.headers.get("Cache-Control", "")
    fresh_ttl, swr_ttl, sie_ttl = _derive_ttls(config, cache_control)
    etag = response.headers.get("ETag")
    if not etag:
        etag = f"\"{hashlib.sha256(content).hexdigest()}\""
    last_modified = response.headers.get("Last-Modified") or datetime.now(dt_timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    cache_dir = get_cache_dir()
    metadata = {
        "path": normalized,
        "key": path_to_cache_key(normalized),
        "url": str(response.url or url),
        "status_code": response.status_code,
        "etag": etag,
        "last_modified": last_modified,
        "generated_at": now,
        "expires_at": now + fresh_ttl,
        "stale_while_revalidate_until": now + fresh_ttl + swr_ttl,
        "stale_if_error_until": now + fresh_ttl + swr_ttl + sie_ttl,
        "fresh_ttl_seconds": fresh_ttl,
        "stale_while_revalidate_seconds": swr_ttl,
        "stale_if_error_seconds": sie_ttl,
        "source_cache_control": cache_control,
        "content_type": response.headers.get("Content-Type", "text/html; charset=utf-8"),
        "content_length": len(content),
        "html_file": str(html_path.relative_to(cache_dir)),
        "metadata_file": str(metadata_path.relative_to(cache_dir)),
        "headers": {
            "Cache-Control": cache_control,
            "ETag": etag,
            "Last-Modified": last_modified,
        },
    }
    _atomic_write_bytes(html_path, content)
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True))
    _update_manifest_entry(normalized, metadata)
    metadata["html_path"] = str(html_path)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _refresh_snapshot_not_modified(path: str, url: str, previous: dict, config: PrerenderConfig) -> dict:
    normalized, _html_path, metadata_path = _snapshot_paths(path)
    now = int(time.time())
    fresh_ttl = _safe_int(previous.get("fresh_ttl_seconds"), config.fresh_ttl_seconds)
    swr_ttl = _safe_int(previous.get("stale_while_revalidate_seconds"), config.stale_while_revalidate_seconds)
    sie_ttl = _safe_int(previous.get("stale_if_error_seconds"), config.stale_if_error_seconds)
    refreshed = {
        **{k: v for k, v in previous.items() if k != "content"},
        "path": normalized,
        "url": previous.get("url") or url,
        "generated_at": now,
        "expires_at": now + max(fresh_ttl, 0),
        "stale_while_revalidate_until": now + max(fresh_ttl, 0) + max(swr_ttl, 0),
        "stale_if_error_until": now + max(fresh_ttl, 0) + max(swr_ttl, 0) + max(sie_ttl, 0),
        "fresh_ttl_seconds": max(fresh_ttl, 0),
        "stale_while_revalidate_seconds": max(swr_ttl, 0),
        "stale_if_error_seconds": max(sie_ttl, 0),
    }
    _atomic_write_text(metadata_path, json.dumps(refreshed, indent=2, sort_keys=True))
    _update_manifest_entry(normalized, refreshed)
    return refreshed


def _fetch_with_retries(
    *,
    url: str,
    headers: dict[str, str],
    timeout_seconds: tuple[int, int],
    retries: int,
    backoff_seconds: float,
) -> tuple[requests.Response | None, str | None]:
    attempts = 1 + max(retries, 0)
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            return response, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    return None, last_error or "request failed"


def prerender_single_path(
    *,
    path: str,
    timeout: int | None = None,
    user_agent: str | None = None,
    retries: int | None = None,
    force: bool = False,
) -> tuple[dict | None, str | None]:
    if not is_prerender_enabled():
        return None, "PRERENDER_ENABLED is false"

    config = get_prerender_config(timeout_override=timeout, user_agent_override=user_agent, retries_override=retries)
    normalized = normalize_path(path, allow_query=True, allowed_query_keys=config.allowed_query_keys)
    previous = None if force else load_prerender_snapshot(normalized)
    url = build_prerender_url(normalized, site_url=config.site_url)

    request_headers = {
        "User-Agent": config.user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "X-Bunoraa-Prerender": "1",
    }
    if previous:
        if previous.get("etag"):
            request_headers["If-None-Match"] = str(previous["etag"])
        if previous.get("last_modified"):
            request_headers["If-Modified-Since"] = str(previous["last_modified"])

    timeout_tuple = (
        max(min(config.connect_timeout_seconds, config.timeout_seconds), 1),
        max(config.read_timeout_seconds, 1),
    )
    response, fetch_error = _fetch_with_retries(
        url=url,
        headers=request_headers,
        timeout_seconds=timeout_tuple,
        retries=config.retries,
        backoff_seconds=config.backoff_seconds,
    )
    if response is None:
        return None, fetch_error

    try:
        if response.status_code == 304 and previous:
            refreshed = _refresh_snapshot_not_modified(normalized, url, previous, config)
            return refreshed, None
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            return None, f"non-HTML content type: {response.headers.get('Content-Type', '')}"
        content = response.content or b""
        if not content:
            return None, "empty response body"
        if len(content) > config.max_content_bytes:
            return None, f"response exceeds PRERENDER_MAX_CONTENT_BYTES ({config.max_content_bytes})"
        metadata = _save_snapshot(normalized, url, response, content, config)
        return metadata, None
    except Exception as exc:
        return None, str(exc)


def prerender_paths(
    *,
    paths: list[str],
    timeout: int = 15,
    user_agent: str = DEFAULT_PRERENDER_USER_AGENT,
    retries: int | None = None,
    force: bool = False,
) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
    saved = 0
    successes: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw_path in paths:
        normalized = normalize_path(raw_path, allow_query=True, allowed_query_keys=get_prerender_config().allowed_query_keys)
        if normalized in seen:
            continue
        seen.add(normalized)
        metadata, error = prerender_single_path(
            path=normalized,
            timeout=timeout,
            user_agent=user_agent,
            retries=retries,
            force=force,
        )
        if metadata:
            output = str(metadata.get("html_path") or "")
            successes.append((normalized, output))
            saved += 1
        else:
            failures.append((build_prerender_url(normalized), error or "unknown prerender error"))

    return saved, successes, failures


def trigger_background_prerender(path: str, *, force: bool = True) -> bool:
    if not is_prerender_enabled():
        return False
    normalized = normalize_path(path, allow_query=True, allowed_query_keys=get_prerender_config().allowed_query_keys)
    key = path_to_cache_key(normalized)
    with _REFRESH_GUARD:
        if key in _REFRESH_INFLIGHT:
            return False
        _REFRESH_INFLIGHT.add(key)

    def _runner():
        try:
            prerender_single_path(path=normalized, force=force)
        except Exception:
            LOGGER.exception("Background prerender failed for %s", normalized)
        finally:
            with _REFRESH_GUARD:
                _REFRESH_INFLIGHT.discard(key)

    thread = threading.Thread(target=_runner, daemon=True, name=f"prerender-{key[:8]}")
    thread.start()
    return True


def is_path_allowed_for_prerender(path: str, patterns: list[str] | tuple[str, ...] | None = None) -> bool:
    normalized = normalize_path(path, allow_query=False)
    active_patterns = (
        patterns
        if patterns is not None
        else getattr(settings, "PRERENDER_PATH_PATTERNS", getattr(settings, "PRERENDER_PATHS", ["/"]))
    )
    for raw_pattern in active_patterns:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        if pattern == "*":
            return True
        wildcard = pattern.endswith("*")
        base = pattern[:-1] if wildcard else pattern
        base_normalized = normalize_path(base, allow_query=False)
        if wildcard:
            prefix = base_normalized.rstrip("/")
            if not prefix:
                return True
            if normalized == f"{prefix}/" or normalized.startswith(f"{prefix}/"):
                return True
            continue
        if normalized == base_normalized:
            return True
    return False
