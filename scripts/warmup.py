import os
import sys
import time
import urllib.request


def ping(url: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BunoraaWarmup/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            return 200 <= status < 400, f"{status}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    raw = os.environ.get(
        "WARMUP_URLS",
        "https://api.bunoraa.com/healthz,https://bunoraa.com/file.svg",
    )
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        print("No WARMUP_URLS configured.")
        return 0

    timeout = int(os.environ.get("WARMUP_TIMEOUT", "10"))
    failures = 0
    for url in urls:
        ok, detail = ping(url, timeout=timeout)
        status = "OK" if ok else "FAIL"
        print(f"{status}: {url} ({detail})")
        if not ok:
            failures += 1
        time.sleep(0.2)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
