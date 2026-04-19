#!/usr/bin/env python3
"""
Bunoraa Warmup Script with Comprehensive Logging
=================================================

This script health-checks application endpoints during startup
to ensure services are ready before accepting traffic.

Environment Variables:
    WARMUP_URLS: Comma-separated list of URLs to ping
    WARMUP_TIMEOUT: Request timeout in seconds (default: 30)
    WARMUP_RETRIES: Number of retries per URL (default: 3)
    WARMUP_RETRY_DELAY_MS: Delay between retries in ms (default: 2000)
    WARMUP_PARALLEL: Enable parallel requests (default: False)
    WARMUP_LOG_LEVEL: Logging level (default: INFO)

Exit Codes:
    0: All endpoints healthy
    1: One or more endpoints failed
    2: Configuration error
"""

import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


# ============================================
# LOGGING SETUP
# ============================================
def setup_logging():
    """Configure structured logging for warmup script."""
    log_level = os.environ.get('WARMUP_LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s [WARMUP] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr,
    )
    return logging.getLogger('warmup')


logger = setup_logging()


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class PingResult:
    """Result of a ping operation."""
    url: str
    success: bool
    status_code: Optional[int] = None
    response_time_ms: float = 0.0
    error_message: str = ""
    attempt: int = 0
    total_attempts: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self):
        return asdict(self)


@dataclass
class WarmupConfig:
    """Warmup configuration."""
    urls: list[str]
    timeout: int
    retries: int
    retry_delay_ms: int
    parallel: bool
    max_workers: int
    output_json: bool


# ============================================
# UTILITY FUNCTIONS
# ============================================
def _with_cache_bust(url: str) -> str:
    """Add cache-busting parameter to URL."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("warmup_ts", str(int(time.time()))))
    query.append(("warmup_id", f"{os.getpid()}_{int(time.time() * 1000)}"))
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def _create_request(url: str) -> urllib.request.Request:
    """Create HTTP request with appropriate headers."""
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "BunoraaWarmup/2.0",
            "Accept": "application/json, text/html, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
        method="GET",
    )


def _format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


# ============================================
# PING OPERATION
# ============================================
def ping_single_attempt(url: str, timeout: int, attempt: int, total_attempts: int) -> PingResult:
    """Execute a single ping attempt."""
    start_time = time.time()
    request_url = _with_cache_bust(url)

    logger.debug('[PING] Attempt %d/%d for %s', attempt, total_attempts, url)

    try:
        req = _create_request(request_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.time() - start_time) * 1000
            status = resp.getcode()
            content_length = resp.headers.get('Content-Length')
            content_type = resp.headers.get('Content-Type', 'unknown')

            if 200 <= status < 400:
                logger.info(
                    '[PING] SUCCESS url=%s status=%d time=%.2fms type=%s size=%s',
                    url, status, elapsed_ms, content_type,
                    _format_bytes(int(content_length)) if content_length else 'unknown'
                )
                return PingResult(
                    url=url,
                    success=True,
                    status_code=status,
                    response_time_ms=elapsed_ms,
                    attempt=attempt,
                    total_attempts=total_attempts,
                )
            else:
                logger.warning(
                    '[PING] UNEXPECTED_STATUS url=%s status=%d time=%.2fms',
                    url, status, elapsed_ms
                )
                return PingResult(
                    url=url,
                    success=False,
                    status_code=status,
                    response_time_ms=elapsed_ms,
                    error_message=f"HTTP {status}",
                    attempt=attempt,
                    total_attempts=total_attempts,
                )

    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            '[PING] HTTP_ERROR url=%s status=%d time=%.2fms error=%s',
            url, e.code, elapsed_ms, str(e)
        )
        return PingResult(
            url=url,
            success=False,
            status_code=e.code,
            response_time_ms=elapsed_ms,
            error_message=f"HTTP {e.code}: {e.reason}",
            attempt=attempt,
            total_attempts=total_attempts,
        )

    except urllib.error.URLError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            '[PING] URL_ERROR url=%s time=%.2fms error=%s reason=%s',
            url, elapsed_ms, type(e).__name__, str(e.reason)
        )
        return PingResult(
            url=url,
            success=False,
            response_time_ms=elapsed_ms,
            error_message=f"{type(e).__name__}: {e.reason}",
            attempt=attempt,
            total_attempts=total_attempts,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.exception(
            '[PING] EXCEPTION url=%s time=%.2fms error=%s',
            url, elapsed_ms, str(e)
        )
        return PingResult(
            url=url,
            success=False,
            response_time_ms=elapsed_ms,
            error_message=f"{type(e).__name__}: {str(e)}",
            attempt=attempt,
            total_attempts=total_attempts,
        )


def ping_with_retry(url: str, timeout: int, retries: int, retry_delay_ms: int) -> PingResult:
    """Ping URL with retry logic."""
    total_attempts = retries + 1
    logger.info('[PING] Starting ping for %s (timeout=%ds, retries=%d)', url, timeout, retries)

    for attempt in range(1, total_attempts + 1):
        result = ping_single_attempt(url, timeout, attempt, total_attempts)

        if result.success:
            logger.info('[PING] SUCCESS for %s after %d attempt(s)', url, attempt)
            return result

        if attempt < total_attempts:
            delay_sec = max(0.5, retry_delay_ms / 1000.0)
            logger.warning(
                '[PING] RETRY url=%s after %.1fs (attempt %d/%d failed: %s)',
                url, delay_sec, attempt, total_attempts, result.error_message
            )
            time.sleep(delay_sec)

    logger.error('[PING] FAILED for %s after %d attempt(s): %s', url, total_attempts, result.error_message)
    return result


# ============================================
# CONFIGURATION
# ============================================
def load_config() -> WarmupConfig:
    """Load configuration from environment variables."""
    raw_urls = os.environ.get(
        "WARMUP_URLS",
        "https://api.bunoraa.com/api/docs/",
    )
    urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

    if not urls:
        logger.error('[CONFIG] No WARMUP_URLS configured')
        raise ValueError("WARMUP_URLS is empty or not configured")

    config = WarmupConfig(
        urls=urls,
        timeout=max(5, int(os.environ.get("WARMUP_TIMEOUT", "30"))),
        retries=max(0, int(os.environ.get("WARMUP_RETRIES", "3"))),
        retry_delay_ms=max(500, int(os.environ.get("WARMUP_RETRY_DELAY_MS", "2000"))),
        parallel=os.environ.get("WARMUP_PARALLEL", "false").lower() == "true",
        max_workers=min(10, max(1, int(os.environ.get("WARMUP_MAX_WORKERS", "3")))),
        output_json=os.environ.get("WARMUP_OUTPUT_JSON", "false").lower() == "true",
    )

    logger.info('[CONFIG] Warmup config: timeouts=%ds, retries=%d, parallel=%s, urls=%d',
                config.timeout, config.retries, config.parallel, len(config.urls))

    return config


# ============================================
# MAIN EXECUTION
# ============================================
def warmup_serial(config: WarmupConfig) -> list[PingResult]:
    """Run warmup checks serially."""
    results = []
    for url in config.urls:
        result = ping_with_retry(url, config.timeout, config.retries, config.retry_delay_ms)
        results.append(result)
        # Small delay between requests to avoid overwhelming
        time.sleep(0.1)
    return results


def warmup_parallel(config: WarmupConfig) -> list[PingResult]:
    """Run warmup checks in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_url = {
            executor.submit(ping_with_retry, url, config.timeout, config.retries, config.retry_delay_ms): url
            for url in config.urls
        }
        for future in as_completed(future_to_url):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                url = future_to_url[future]
                logger.exception('[PING] Unexpected error for %s: %s', url, exc)
                results.append(PingResult(url=url, success=False, error_message=str(exc)))
    return results


def print_results(results: list[PingResult], output_json: bool = False):
    """Print results in requested format."""
    if output_json:
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total": len(results),
            "success": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "="*70)
        print("WARMUP SUMMARY")
        print("="*70)
        for r in results:
            status = "✓ OK" if r.success else "✗ FAIL"
            time_str = f"{r.response_time_ms:.2f}ms" if r.response_time_ms > 0 else "N/A"
            print(f"{status}: {r.url}")
            if r.success:
                print(f"       Status: {r.status_code} | Time: {time_str}")
            else:
                print(f"       Error: {r.error_message}")
        print("="*70)
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes
        print(f"Total: {len(results)} | Success: {successes} | Failed: {failures}")


def main() -> int:
    """Main entry point."""
    start_time = time.time()
    logger.info('[START] Bunoraa Warmup Script v2.0 (pid=%d)', os.getpid())

    try:
        config = load_config()
    except ValueError as e:
        logger.error('[CONFIG] %s', e)
        return 2

    # Execute warmup
    if config.parallel:
        logger.info('[EXECUTE] Running warmup in parallel mode with %d workers', config.max_workers)
        results = warmup_parallel(config)
    else:
        logger.info('[EXECUTE] Running warmup in serial mode')
        results = warmup_serial(config)

    # Calculate final results
    failures = sum(1 for r in results if not r.success)
    total_time = time.time() - start_time

    # Print results
    print_results(results, config.output_json)

    # Log summary
    if failures == 0:
        logger.info(
            '[COMPLETE] All endpoints healthy in %.2fs (total_time=%.2fs)',
            total_time, total_time
        )
    else:
        logger.warning(
            '[COMPLETE] %d/%d endpoints failed in %.2fs',
            failures, len(results), total_time
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
