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

## Adding a new GPU (runbook)

Worked example pattern used for all 19 current GPUs. Do steps 1–2 (discovery)
before writing any code.

### 1. Find the SHS slug

```bash
curl -s -A "Mozilla/5.0" "https://secondhandsilicon.com/api/search?q=rtx+4080" \
  | python3 -c "import sys,json; [print(d['id'], d['basePrice']) for d in json.load(sys.stdin)[:8]]"
```

Pick the exact-match `id` (ignore `isFuzzyMatch` neighbours). Note `basePrice`
for later comparison with TPU's launch price. **If no slug exists** (e.g. RTX
3050 6GB), stop: the GPU stays out until price coverage exists — no price,
no ranking.

### 2. Find the TPU reference page (headed Chromium only)

```bash
playwright-cli -s=tpuheaded open --headed "https://www.techpowerup.com/gpu-specs/"
# then list reference links matching your card, e.g.:
playwright-cli -s=tpuheaded eval "() => [...document.querySelectorAll('a')].map(a=>[a.textContent.trim(),a.getAttribute('href')]).filter(([t,h])=>/GeForce RTX 4080/.test(t)&&h&&h.includes('/gpu-specs/')).map(([t,h])=>t+' -> '+h).filter((v,i,a)=>a.indexOf(v)===i).join('\n')" --raw
```

The numeric suffix (`cXXXX`) is the `tpu_id`. **If no reference page exists**
(quiet AIB-only launches — verified before in the live table, not assumed),
follow the sibling policy in `docs/METHODOLOGY.md`: `tpu_url: null`, inherit a
sibling SKU, memory subsystem from TPU's own reporting, uncertain fields
flagged in comments. Session name must stay `tpuheaded` (the fetch script
depends on it); the script `cd`s to scratch so no session files pollute the repo.

### 3. Append `config.py`

```python
{
    "id": "rtx-4080",                    # our slug: series-model-variant, lowercase
    "name": "NVIDIA GeForce RTX 4080",   # full display name
    "short_name": "RTX 4080",            # leaderboard label
    "tpu_url": "https://www.techpowerup.com/gpu-specs/geforce-rtx-4080.cXXXX",
    "tpu_id": "cXXXX",                   # None when no reference page exists
    "shs_slug": "nvidia-rtx-4080",
    "shs_url": "https://secondhandsilicon.com/product/nvidia-rtx-4080",
    # "relperf_name": "GeForce RTX 4080",  # ONLY if TPU's gaming-table name
},                                       # differs (SUPER caps, "12 GB" spaced…)
```

Relperf aliases must be verified against the `relperf` table
(`SELECT name FROM relperf`), never guessed — a wrong alias attributes another
card's score. Cards absent from TPU's table stay `null` (no fallback).

### 4. Add a `tpu_curated.py` row

Copy a same-architecture row and adjust. Whitepaper/formula values are
acceptable (Tensor ratios by generation, `FP32 = CUDA × boost × 2`);
**mark every guess in a comment** — the live diff in step 6 is authoritative
and will list each mismatch as `DIFF vs curated`. New architecture with FP4?
Stop and implement `max()` precision selection first (`docs/ROADMAP.md`).

### 5. Dump, then crawl the new cards first

```bash
# extend the URLS map in tools/fetch_tpu_live.sh, then:
./tools/fetch_tpu_live.sh /tmp/tpu_new          # ~7 s politeness gap per page
.venv/bin/python -m scraper.run --only rtx-4080 --tpu-live-dir /tmp/tpu_new
```

Read the output: fix every `DIFF vs curated` line (live wins, update curated),
check the price flag/`n`, confirm `tpu_method` is `live_headed` (not
`curated`/`sibling` unless intended).

### 6. Verify, then commit

```bash
.venv/bin/python - <<'PY'   # relperf + status audit
import sys; sys.path.insert(0,'.'); from dbtools import query; from pathlib import Path
con = query.open_ro(Path('gpu.db'))
r = query.get_gpu_full(con, 'rtx-4080')
print(r['relperf_name'], r['status'], r['ai'], r['pricing']['used']['canon'])
print('integrity:', con.execute('PRAGMA integrity_check').fetchone()[0])
PY
curl -s localhost:8000/api/gpus/rtx-4080 | python3 -m json.tool | head -20
./serve 8000  # eyeball /website/ (row count, leader, footer stamp)
# gpu.db is gitignored by design (data stays local); commit code + docs only:
git add scraper/config.py scraper/tpu_curated.py tools/fetch_tpu_live.sh
git commit -m "feat: add RTX 4080 coverage" && git push
```

`gpu.db` is gitignored by design (data stays local); the commit carries code +
docs only. A full (non-`--only`) run afterwards refreshes everything.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `FirewallBlocked` for all GPUs | No dump dir and datacenter IP → dump via headed Chromium first |
| Served leaderboard shrinks after `--only` | Historical bug (missing row factory dropped DB rows from summary merge) — fixed; summaries always cover full DB state |
| `history +N` on identical data | Change-hash inputs changed (frozen-form lesson) → one-time re-baseline, then `+0` |
| `-shm`/`-wal` files next to `gpu.db` | Un-checkpointed WAL → runs checkpoint on commit; `VACUUM` in maint |
| `.playwright-cli/` in repo | Browser session ran with repo as CWD → delete; fetch script cds to scratch |
| `Cannot read properties of null (reading 'price_usd_…')` | GPU with failed price fetch — fixed: fetches never wipe previous rows, site isolates unpriced cards, crawl exits 2 with retry command |
| `…(reading 'toFixed')` or literal `null`/`NaN` on cards | Pre-Tensor cards lack matrix rows; old cards lack clocks/MSRP — fixed: null-preserving loader math, `F1/F2/F3/FM` formatters, `?? "—"` on spec interpolations |
| `record_sha` edits | **Frozen discipline**: changing its canonical form re-baselines ALL history — treat as a migration, not a tweak |
| `database is locked` on crawl | Another crawl is mid-commit (WAL checkpoint holds EXCLUSIVE) → never run two crawls concurrently; never `kill -9` a committing process (can orphan WAL frames); wait for the lock holder, then verify row counts |
| Background crawl "completes" but rows missing | Killed mid-checkpoint (see above) → re-crawl the batch; `history` dedup makes reruns cheap |

## Scale-up lessons (bulk expansion runs)

- **TPU table renders ~101 recent rows only** — older/workstation cards are
  invisible there. Resolve their IDs via websearch snippets (canonical
  `.../gpu-specs/<slug>.cXXXX` URLs), then verify each with a headed visit
  (title must match) before adding to `config.py`.
- **TPU unit traps** (all fixed in `tpu_live.py`, keep them in mind for new
  fields): bandwidth `TB/s` (×1000); Theoretical FP16/FP32/FP64 cells mix
  TFLOPS/GFLOPS by card age; Memory Size in MB pre-2012; transistors in
  millions for sub-billion dice; HBM memory clocks in Mbps.
- **Variant entries sharing one TPU page** (4GB/8GB twins, dual-GPU boards):
  use `tpu_overrides` in `config.py` for entry-defining fields so live-wins
  doesn't erase the variant (memory size, board-total compute). Overrides
  participate in the live-vs-curated diff, so matching curated values stay
  DIFF-free.
- **SHS throttling**: batches of ~20 GPUs (80 calls) take several minutes;
  run crawls in background, one at a time, and expect occasional
  `TimeoutError -> missing` conditions (retry on the next full run).

## Scheduling

No scheduler is configured. Sensible default when wanted: weekly full crawl
(SHS revises daily; TPU specs ~never) via cron, then `./serve` keeps serving —
no rebuild step exists. Keep an eye on SHS call volume when scaling SKUs.
