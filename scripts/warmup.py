import os
import sys
import time
import urllib.parse
import urllib.request


def _with_cache_bust(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("warmup_ts", str(int(time.time()))))
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def ping(url: str, timeout: int = 45, retries: int = 2, retry_delay_ms: int = 1200) -> tuple[bool, str]:
    last_error = "unknown_error"
    attempts = max(1, retries + 1)

    for attempt in range(1, attempts + 1):
        request_url = _with_cache_bust(url)
        try:
            req = urllib.request.Request(
                request_url,
                headers={
                    "User-Agent": "BunoraaWarmup/1.1",
                    "Cache-Control": "no-cache, no-store, max-age=0",
                    "Pragma": "no-cache",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.getcode()
                if 200 <= status < 400:
                    return True, f"{status} (attempt {attempt}/{attempts})"
                last_error = f"HTTP {status} (attempt {attempt}/{attempts})"
        except Exception as exc:
            last_error = f"{exc} (attempt {attempt}/{attempts})"

        if attempt < attempts:
            time.sleep(max(0, retry_delay_ms) / 1000.0)

    return False, last_error


def main() -> int:
    raw = os.environ.get(
        "WARMUP_URLS",
        "https://bunoraa-django.onrender.com/health/,https://bunoraa-pl26.onrender.com/,https://api.bunoraa.com/health/",
    )
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        print("No WARMUP_URLS configured.")
        return 0

    timeout = max(1, int(os.environ.get("WARMUP_TIMEOUT", "45")))
    retries = max(0, int(os.environ.get("WARMUP_RETRIES", "2")))
    retry_delay_ms = max(0, int(os.environ.get("WARMUP_RETRY_DELAY_MS", "1200")))

    print(
        f"Warmup targets={len(urls)} timeout={timeout}s retries={retries} retry_delay_ms={retry_delay_ms}"
    )

    failures = 0
    for url in urls:
        ok, detail = ping(
            url,
            timeout=timeout,
            retries=retries,
            retry_delay_ms=retry_delay_ms,
        )
        status = "OK" if ok else "FAIL"
        print(f"{status}: {url} ({detail})")
        if not ok:
            failures += 1
        time.sleep(0.2)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
