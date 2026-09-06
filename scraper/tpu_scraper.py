"""TechPowerUp fetch layer: plain HTTP attempt + firewall detection.

What actually works (verified 2026-09-06):
  - Plain HTTP (curl/urllib): gets the PoW/drag-captcha challenge page.
  - Headless Chromium: hard `403 Access Denied` (automation fingerprint).
  - Headed Chromium via `playwright-cli`: passes. Dumps happen through
    `tools/fetch_tpu_live.sh`; parsing lives in `scraper/tpu_live.py`.

This module remains as the programmatic fallback path used by
`scraper/run.py::load_live` when no dumped page is available.
"""

from __future__ import annotations

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

FIREWALL_MARKERS = [
    "Automated bot check in progress",
    "pow-progress-bar",
    "drag-captcha",
    "/.firewall",
]


class FirewallBlocked(Exception):
    pass


def is_firewall_page(html: str) -> bool:
    return any(m in html for m in FIREWALL_MARKERS)


def fetch_html_requests(url: str, timeout: int = 30) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_html_playwright(url: str, timeout_ms: int = 60000) -> str:
    """Headless-Chromium attempt (usually 403s; headed CLI is the proven path)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=BROWSER_UA)
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # Give the PoW challenge time to solve itself.
        page.wait_for_timeout(8000)
        html = page.content()
        browser.close()
        return html


def fetch_tpu_html(url: str) -> str:
    """Best-effort fetch: plain HTTP first, Playwright fallback."""
    html = fetch_html_requests(url)
    if not is_firewall_page(html):
        return html
    try:
        html2 = fetch_html_playwright(url)
    except Exception as e:  # Playwright not installed / still blocked
        raise FirewallBlocked(
            f"TPU firewall blocked plain HTTP for {url} and Playwright fallback failed: {e}"
        ) from e
    if is_firewall_page(html2):
        raise FirewallBlocked(f"TPU firewall still present for {url} after Playwright")
    return html
