"""SecondHandSilicon price client (v2).

Public JSON API (same one the React frontend uses):
  GET /api/prices/{slug}?period=3y&condition={used|new}   -> stats + monthly buckets
  GET /api/export/{slug}?format=json&condition={used|new}&period=3y -> sold listings

Raw listings are returned (not discarded) so the DB layer can archive them and
compute arbitrary windows._STATS: stats.currentPrice/average == trailing-30d mean.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

API_BASE = "https://secondhandsilicon.com/api"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) gpu-flops-per-dollar/2.0"}
WINDOWS = (7, 30, 90, 365)


def _get_json(url: str, timeout: int = 90) -> dict | list:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_summary(slug: str, condition: str) -> dict:
    return _get_json(f"{API_BASE}/prices/{slug}?period=3y&condition={condition}")


def fetch_listings(slug: str, condition: str) -> list[dict]:
    d = _get_json(f"{API_BASE}/export/{slug}?format=json&condition={condition}&period=3y")
    return d.get("listings", []) if isinstance(d, dict) else []


def window_stats(dates: list[str], prices: list[float], today: dt.date, days: int) -> dict:
    cutoff = (today - dt.timedelta(days=days)).isoformat()
    sel = sorted(p for d, p in zip(dates, prices) if d >= cutoff)
    if not sel:
        return {"n": 0, "avg": None, "med": None, "lo": None, "hi": None}
    n = len(sel)
    med = sel[n // 2] if n % 2 else (sel[n // 2 - 1] + sel[n // 2]) / 2
    return {"n": n, "avg": round(sum(sel) / n, 2), "med": round(med, 2),
            "lo": round(sel[0], 2), "hi": round(sel[-1], 2)}


def fetch_condition(slug: str, condition: str, today: dt.date | None = None) -> dict:
    """Everything SHS knows for one slug+condition, listings included."""
    today = today or dt.date.today()
    summary = fetch_summary(slug, condition)
    raw = fetch_listings(slug, condition)
    dates = [str(l.get("date", "")) for l in raw]
    prices = [float(l["price"]) for l in raw if l.get("price")]
    titles = [str(l.get("title", "")) for l in raw]
    stats = summary.get("stats", {})
    return {
        "summary": summary,
        "stats": stats,
        "api_avg_30d": stats.get("average") or stats.get("currentPrice"),
        "dates": dates,
        "prices": prices,
        "titles": titles,
        "windows": {f"{w}d": window_stats(dates, prices, today, w) for w in WINDOWS},
        "monthly": summary.get("monthlyData", []),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
