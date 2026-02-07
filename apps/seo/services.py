import os
import time
import requests
from bs4 import BeautifulSoup
from django.utils import timezone
from .models import Keyword, SERPSnapshot


USER_AGENT = os.environ.get('SEO_USER_AGENT', 'Mozilla/5.0 (compatible; BunoraaSEO/1.0; +https://bunoraa.com)')


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