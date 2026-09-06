"""Live server: static site + JSON API backed directly by gpu.db.

No build step: the site and all API responses read the current database on
every request. Crawl, then just (re)load the page.

    .venv/bin/python tools/serve.py [port] [--db PATH]

API (all GET, JSON; see docs/API.md):
    /api               index of endpoints
    /api/health        db status, gpu count, last run
    /api/summary       leaderboard + methodology + decisions
    /api/gpus          ranked GPU rows (same as summary.gpus)
    /api/gpus/<id>     full friendly record
    /api/gpus/<id>/prices[?cond=used|new|all]
    /api/gpus/<id>/listings?cond=used|new[&limit=&offset=]
    /api/gpus/<id>/raw  archived TPU HTML (large)
    /api/relperf       shared TPU gaming relative-performance table
    /api/runs          run manifests
    /api/bundle        compact site bundle (what the page uses live)
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import json
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dbtools import query  # noqa: E402  (needs stdlib zstandard via .venv)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Handler(http.server.SimpleHTTPRequestHandler):
    db_path: Path = ROOT / "gpu.db"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    # -- helpers ---------------------------------------------------------
    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _err(self, status: int, msg: str):
        self._send_json({"error": msg}, status)

    def end_headers(self):
        if not self.path.startswith("/api"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # -- routing ---------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api" or parsed.path == "/api/":
            return self._send_json({"endpoints": [
                "/api/health", "/api/summary", "/api/gpus", "/api/gpus/<id>",
                "/api/gpus/<id>/prices?cond=used|new|all",
                "/api/gpus/<id>/listings?cond=used|new&limit&offset",
                "/api/gpus/<id>/raw", "/api/relperf", "/api/runs", "/api/bundle"]})
        if parsed.path.startswith("/api/"):
            return self._api(parsed)
        return super().do_GET()

    do_HEAD = do_GET

    def _api(self, parsed):
        parts = [p for p in parsed.path.split("/") if p][1:]  # drop 'api'
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            con = query.open_ro(self.db_path)
        except Exception as e:
            return self._err(500, f"cannot open db: {e}")
        try:
            if parts == ["health"]:
                runs = query.get_runs(con)
                return self._send_json({
                    "ok": True, "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "gpus": len(query.list_ids(con)),
                    "last_run": runs[-1]["ts"] if runs else None,
                    "runs": len(runs)})
            if parts == ["summary"]:
                return self._send_json(query.get_summary(con))
            if parts == ["gpus"]:
                s = query.get_summary(con)
                return self._send_json({"generated_at": s.get("generated_at"),
                                        "gpus": s.get("gpus", [])})
            if parts == ["relperf"]:
                return self._send_json(query.get_relperf(con))
            if parts == ["runs"]:
                return self._send_json({"runs": query.get_runs(con)})
            if parts == ["bundle"]:
                return self._send_json(_bundle(con))
            if len(parts) >= 2 and parts[0] == "gpus":
                return self._gpu(con, parts[1:], qs)
            return self._err(404, "unknown endpoint (see GET /api)")
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")
        finally:
            con.close()

    def _gpu(self, con, parts, qs):
        gid = parts[0]
        if not ID_RE.match(gid):
            return self._err(400, "bad gpu id")
        if len(parts) == 1:
            r = query.get_gpu_full(con, gid)
            return self._send_json(r) if r else self._err(404, f"unknown gpu: {gid}")
        if len(parts) == 2 and parts[1] == "prices":
            r = query.get_gpu_full(con, gid)
            if not r:
                return self._err(404, f"unknown gpu: {gid}")
            cond = (qs.get("cond") or ["all"])[0]
            pr = r["pricing"]
            if cond == "all":
                return self._send_json({"id": gid, "used": pr["used"], "new": pr["new"]})
            if cond in ("used", "new"):
                return self._send_json({"id": gid, "cond": cond, "data": pr[cond]})
            return self._err(400, "cond must be used|new|all")
        if len(parts) == 2 and parts[1] == "listings":
            cond = (qs.get("cond") or ["used"])[0]
            if cond not in ("used", "new"):
                return self._err(400, "cond must be used|new")
            d = query.get_listings(con, gid, cond)
            total = len(d["rows"])
            try:
                limit = int((qs.get("limit") or [0])[0])
                offset = int((qs.get("offset") or [0])[0])
            except ValueError:
                return self._err(400, "limit/offset must be ints")
            rows = d["rows"][offset:(offset + limit) if limit else None]
            return self._send_json({"id": gid, "cond": cond, "d0": d["d0"],
                                    "total": total, "offset": offset,
                                    "limit": limit or total,
                                    "row": ["date", "price_usd", "title"], "rows": rows})
        if len(parts) == 2 and parts[1] == "raw":
            row = con.execute("SELECT html, sha FROM raw_pages WHERE gpu_id=?",
                              (gid,)).fetchone()
            if not row:
                return self._err(404, f"no raw page for {gid}")
            from dbtools.store import zdecomp_bytes  # noqa
            return self._send_json({"id": gid, "sha": row[1], "html": zdecomp_bytes(row[0])})
        return self._err(404, "unknown endpoint (see GET /api)")


def _bundle(con):
    from dbtools.pack import Pool, compact_record
    pool = Pool()
    ids = query.list_ids(con)
    recs = [query.get_gpu_full(con, gid) for gid in ids]
    return {"v": 2, "pool": pool.items,
            "gpus": [compact_record(r, pool) for r in recs],
            "summary": query.get_summary(con)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve site + live gpu.db API")
    ap.add_argument("port", nargs="?", type=int, default=8000)
    ap.add_argument("--db", default=str(ROOT / "gpu.db"))
    ap.add_argument("--open", action="store_true", help="open /website/ in a browser")
    args = ap.parse_args()
    Handler.db_path = Path(args.db)
    if not Handler.db_path.exists():
        sys.exit(f"db not found: {Handler.db_path}")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/website/"
    print(f"Site  {url}")
    print(f"API   http://localhost:{args.port}/api  (db: {Handler.db_path})")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
