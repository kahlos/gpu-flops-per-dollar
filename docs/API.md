# Live API reference (for humans and agents)

The server reads `gpu.db` on **every request** — no build step, no stale data.
Start it:

```bash
.venv/bin/python tools/serve.py [port]   # default 8000
```

- Site: `http://localhost:8000/website/` (live-only; requires the server).
- Base URL for everything below: `http://localhost:8000/api`
- All responses are JSON (`Access-Control-Allow-Origin: *`, `Cache-Control:
  no-store`). Errors are `{"error": "..."}` with an HTTP status.
- Money is in **dollars** (float), tensor rates in **TFLOPS/TOPS** (float) —
  no schema knowledge needed. (The `.db` stores cents/centi-ints; the API
  converts; see `docs/SCHEMA.md` + `dbtools/query.py`.)

## Endpoints

| Method + path | Description |
|---|---|
| `GET /api` | Index of endpoints |
| `GET /api/health` | `{ok, time, gpus, last_run, runs}` |
| `GET /api/summary` | Leaderboard rows + methodology + decisions + sources |
| `GET /api/gpus` | `{generated_at, gpus: [...]}` ranked rows (`id, short_name, ai_tflops, price_usd, price_new_usd, price_flag, price_n30d, tflops_per_dollar, memory_gb, bandwidth_gbs, tdp_w, bus_interface`) |
| `GET /api/gpus/<id>` | Full record: specs (30 fields), theoretical, matrix, ai, pricing used+new (windows, monthlies, stats), depreciation, status, provenance |
| `GET /api/gpus/<id>/prices?cond=used\|new\|all` | Pricing blocks only (`cond` defaults to `all`) |
| `GET /api/gpus/<id>/listings?cond=used\|new&limit&offset` | Raw sales: `{d0, total, offset, limit, row: ["date","price_usd","title"], rows: [...]}`. Omit `limit` for the full dump |
| `GET /api/gpus/<id>/raw` | Archived TPU spec HTML + sha (large, ~300 KB) |
| `GET /api/relperf` | Shared TPU gaming relative-performance table |
| `GET /api/runs` | Run manifests `{ts, manifest: {gpu_id: record_sha}}` |
| `GET /api/bundle` | Compact site bundle `{v, pool, gpus, summary}` (internal; prefer the endpoints above) |

GPU ids look like `rtx-3090`, `rtx-3080-10gb`, `rtx-3060-12gb` (see `/api/gpus`).

## Examples

```bash
# leaderboard
curl -s localhost:8000/api/gpus | python3 -m json.tool | head -30

# full RTX 3090 record
curl -s localhost:8000/api/gpus/rtx-3090 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ai'], d['specs']['bus_if'])"

# used listings, second page of 5
curl -s "localhost:8000/api/gpus/rtx-3090/listings?cond=used&limit=5&offset=5"

# new-condition 30-day window only (jq)
curl -s "localhost:8000/api/gpus/rtx-3070/prices?cond=new" | jq .data.windows
```

```python
import json, urllib.request
BASE = "http://localhost:8000/api"
def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.load(r)
lb = get("/gpus")["gpus"]                       # ranked rows
r = get("/gpus/rtx-3080-10gb")                  # full record
print(r["ai"])                                  # {t, ts, prec, pd, pds}
sales = get("/gpus/rtx-3080-10gb/listings?cond=used")
print(len(sales["rows"]), "used sales,", sales["rows"][-1])
```
