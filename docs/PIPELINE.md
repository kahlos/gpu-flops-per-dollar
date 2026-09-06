# Pipeline

End-to-end crawl anatomy: sources → `gpu.db` → site/API. Code: `scraper/`,
`dbtools/`; entry point `scraper/run.py`; serve via `tools/serve.py` / `./serve`.

```
config.py (GPU list)
  ├─ TPU: headed-Chromium dump ──▶ parse/normalize (tpu_live.py)
  │        tools/fetch_tpu_live.sh → /tmp/tpu_live/*.html
  └─ SHS API (plain HTTPS): used+new summaries + raw listings (shs_scraper.py)
         │
         ▼
  build_expanded() → store.commit_run() → gpu.db
         │                                     ├─ gpus/monthly/windows/cond_stats (replace)
         │                                     ├─ listing_blobs/raw_pages (zstd, sha-skipped if same)
         │                                     ├─ history (deltas for changed GPUs only)
         │                                     └─ runs (manifest + leaderboard summary)
         ▼
  serve.py ──▶ /website (live /api/bundle) + /api/* (docs/API.md)
```

## TPU fetching (the hard source)

| Client | Outcome |
|---|---|
| Plain HTTP (curl/urllib) | PoW + drag-captcha challenge page |
| Headless Chromium | Hard `403 Access Denied` (automation fingerprint) |
| **Headed Chromium (`playwright-cli -s=tpuheaded open --headed`)** | **Passes — the only working path** |

- Dump: `./tools/fetch_tpu_live.sh [outdir]` (default `/tmp/tpu_live`), ~7 s
  politeness gap between pages. Session must exist first (see command above).
- **Hygiene:** `playwright-cli` writes session files to its **CWD** — the fetch
  script `cd`s to scratch so `.playwright-cli/` never lands in the repo. If it
  does, delete it. Never commit it.
- Parser (`tpu_live.py`) quirks learned the hard way: page text contains
  **literal backslash-escapes** (`\n`, `\t` as characters — normalize them, don't
  just split whitespace); bandwidth ≥1000 GB/s renders as **TB/s** (×1000);
  class attributes contain escaped quotes (match with regex, not CSS classes).
- Specs are static: each GPU's page is fetched **once**, archived in
  `raw_pages`, and reused. Pricing refetches every crawl.

## SHS fetching (the easy source)

Public JSON API, plain HTTPS, no browser needed:

- `GET /api/prices/{slug}?period=3y&condition={used|new}` → stats + monthly buckets
- `GET /api/export/{slug}?format=json&condition={used|new}&period=3y` → raw sold listings
- Slugs follow `nvidia-rtx-4090`, `amd-rx-7900-xtx`; discover via `/api/search?q=`.
- Volume: 4 calls/GPU/crawl (used+new × summary+export). `stats.average` is the
  trailing-30d mean — verified against independent recomputation (<1%).

## Run flags (`python -m scraper.run`)

| Flag | Effect |
|---|---|
| `--tpu-live-dir DIR` | Fresh dumps; required for NEW GPUs or with `--refresh-specs` |
| `--refresh-specs all\|id,..` | Force fresh TPU specs (default: reuse DB copy) |
| `--only id,..` | Pricing-refresh subset (summary still covers the full DB) |
| `--db PATH` | Target database (default `gpu.db`; use a copy for experiments) |

Storage rules per run: pricing tables are replaced wholesale (PKs prevent
dupes); listing/raw blobs skip byte-identical rewrites; history appends only
for data-changed GPUs (timestamps excluded from change hashes — reruns with
identical data write nothing but the run manifest); relperf table is only
replaced when fresh rows exist (never wiped).

## Adding a new GPU (checklist)

1. **SHS slug**: `/api/search?q=` → confirm slug + note `basePrice`.
2. **TPU page**: check the live GPU Database table for a *reference* page
   (AIB-only launches have none — use the sibling policy, `tpu_url: null`).
3. **config.py**: append entry (`id, name, short_name, tpu_url, tpu_id,
   shs_slug, shs_url`, plus `relperf_name` alias if TPU's table naming differs —
   verify against the `relperf` table, never guess).
4. **tpu_curated.py**: add baseline row (whitepaper/formula values acceptable;
   mark guesses — the live diff is authoritative).
5. **Dump**: `./tools/fetch_tpu_live.sh` (extend its URL map), then run with
   `--tpu-live-dir`. Confirm zero `tpu_discrepancies`, or fix curated.
6. **Verify**: `relperf_name` non-null (unless genuinely absent), price flag sane,
   `PRAGMA integrity_check`, API smoke (`/api/gpus/<id>`), site renders.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `FirewallBlocked` for all GPUs | No dump dir and datacenter IP → dump via headed Chromium first |
| Served leaderboard shrinks after `--only` | Historical bug (missing row factory dropped DB rows from summary merge) — fixed; summaries always cover full DB state |
| `history +N` on identical data | Change-hash inputs changed (frozen-form lesson) → one-time re-baseline, then `+0` |
| `-shm`/`-wal` files next to `gpu.db` | Un-checkpointed WAL → runs checkpoint on commit; `VACUUM` in maint |
| `.playwright-cli/` in repo | Browser session ran with repo as CWD → delete; fetch script cds to scratch |
| `record_sha` edits | **Frozen discipline**: changing its canonical form re-baselines ALL history — treat as a migration, not a tweak |

## Scheduling

No scheduler is configured. Sensible default when wanted: weekly full crawl
(SHS revises daily; TPU specs ~never) via cron, then `./serve` keeps serving —
no rebuild step exists. Keep an eye on SHS call volume when scaling SKUs.
