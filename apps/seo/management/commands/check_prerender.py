import time
from urllib.parse import urljoin

import requests
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.seo.services import (
    build_prerender_url,
    get_prerender_config,
    is_prerender_enabled,
    list_prerender_entries,
    normalize_path,
)


class Command(BaseCommand):
    help = (
        "Validate prerender snapshots and bot-serving behavior "
        "(X-PreRendered headers, freshness state, and TTFB budget)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-ttfb-ms",
            type=int,
            default=800,
            help="Maximum acceptable bot TTFB in milliseconds",
        )
        parser.add_argument(
            "--strict-fresh",
            action="store_true",
            help="Fail entries that are not in fresh state",
        )

    def handle(self, *args, **options):
        if not is_prerender_enabled():
            self.stdout.write(self.style.WARNING("PRERENDER_ENABLED is false; skipping prerender check."))
            return

        entries = list_prerender_entries()
        if not entries:
            self.stdout.write(self.style.WARNING("No prerender snapshots found in manifest."))
            return

        max_ttfb_ms = max(int(options.get("max_ttfb_ms") or 800), 50)
        strict_fresh = bool(options.get("strict_fresh"))
        config = get_prerender_config()
        site_url = getattr(settings, "SITE_URL", config.site_url)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }

        passed = 0
        failed = 0
        total = len(entries)

        for entry in entries:
            path = normalize_path(str(entry.get("path") or "/"), allow_query=True, allowed_query_keys=config.allowed_query_keys)
            url = build_prerender_url(path, site_url=site_url)
            status = "FAIL"
            reason = "unknown"
            try:
                start = time.perf_counter()
                response = requests.get(urljoin(str(site_url).rstrip("/") + "/", path.lstrip("/")), headers=headers, timeout=10)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                prerendered = response.headers.get("X-PreRendered") == "1"
                state = response.headers.get("X-PreRendered-State", "").strip().lower()
                state_ok = state == "fresh" or (state in {"stale-while-revalidate", "stale-if-error"} and not strict_fresh)
                ttfb_ok = elapsed_ms <= max_ttfb_ms
                http_ok = response.status_code == 200
                if http_ok and prerendered and state_ok and ttfb_ok:
                    status = "OK"
                    reason = f"{response.status_code} {elapsed_ms}ms state={state or 'unknown'}"
                    passed += 1
                else:
                    checks = []
                    if not http_ok:
                        checks.append(f"http={response.status_code}")
                    if not prerendered:
                        checks.append("missing X-PreRendered")
                    if not state_ok:
                        checks.append(f"state={state or 'missing'}")
                    if not ttfb_ok:
                        checks.append(f"ttfb={elapsed_ms}ms>{max_ttfb_ms}ms")
                    reason = ", ".join(checks) if checks else "validation failed"
                    failed += 1
            except Exception as exc:
                reason = str(exc)
                failed += 1

            formatter = self.style.SUCCESS if status == "OK" else self.style.ERROR
            self.stdout.write(formatter(f"{status}: {url} -> {reason}"))

        summary = f"Checked {total} snapshots: {passed} passed, {failed} failed."
        if failed:
            self.stdout.write(self.style.WARNING(summary))
        else:
            self.stdout.write(self.style.SUCCESS(summary))
