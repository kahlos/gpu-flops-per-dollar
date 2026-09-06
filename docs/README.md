# GPU FLOPS per Dollar

**Aim:** rank AI accelerators by expected low-precision tensor throughput per
used dollar — dense `max(FP4, INT4)` from TechPowerUp, divided by the
trailing-30-day used price from SecondHandSilicon (eBay sold listings) — plus
the memory, bandwidth, power and interface context needed to judge real
deployability. Full rules: `docs/METHODOLOGY.md`.

**Status:** 157 GPUs in `gpu.db` — GeForce GTX 400 → RTX 50 (incl. dual-GPU
boards + all Titans), Radeon HD 5000 → RX 9000 (incl. R9 Fury/Nano-era
flagships), Intel Arc A/B, Quadro K/M/P/RTX + RTX Ax000 workstation cards,
Tesla K/M/P/V/A100 datacenter cards, Radeon Pro W7800. Leader at last crawl:
RTX 2080 Super, 1.62 TFLOPS/$ (live values: `./serve`, then `/api/gpus`, or
`SELECT short_name, ai_pd/10000.0 FROM gpus ORDER BY ai_pd DESC` — never trust a
frozen table when the database is right there).

## Layout (database + docs + code, nothing else)

```
gpu.db          THE database (SQLite + zstd-22, ~1 MB for 19 GPUs)
serve           one-command launcher: site + JSON API from gpu.db
docs/           README.md (this index), METHODOLOGY.md, PIPELINE.md,
                SCHEMA.md (table reference), API.md, ROADMAP.md,
                CLUSTER-VIABILITY-ANALYSIS.md (single consolidated research
                doc: model spec, cluster economics vs API, research history)
scraper/        crawl code: config, TPU parsers, SHS client, curated baseline, run.py
dbtools/        database code: schema.sql, store, query, delta, pack, maint
website/        static site, live-only (reads /api/bundle from gpu.db)
tools/          fetch_tpu_live.sh (headed-Chromium dumps), serve.py (server impl),
                cluster_model.py (self-checking cluster economics model)
requirements.txt
```

## Usage

```bash
./serve [port] [--open]   # site at /website/, JSON API at /api/ (docs/API.md)
.venv/bin/python -m scraper.run                  # pricing refresh; specs reused from DB
.venv/bin/python -m scraper.run --only rtx-3090  # pricing subset (summary stays full-DB)
.venv/bin/python -m scraper.run --tpu-live-dir /tmp/tpu_live  # + fresh TPU dumps
.venv/bin/python -m scraper.run --refresh-specs all --tpu-live-dir /tmp/tpu_live
```

Analyze from Python (transparent decompression, friendly units):

```python
import sys; sys.path.insert(0, '.')
from pathlib import Path
from dbtools import query
con = query.open_ro(Path('gpu.db'))
query.leaderboard(con)                  # ranked TFLOPS/$ rows
query.get_gpu_full(con, 'rtx-3090')     # full record
query.get_listings(con, 'rtx-3090', 'used')['rows']  # [[date, price, title]]
```

## Docs for future agents

- `METHODOLOGY.md` — metric definition, locked decisions, provenance + coverage rules, limitations.
- `PIPELINE.md` — crawl anatomy, TPU firewall, run flags, new-GPU checklist, troubleshooting, scheduling.
- `SCHEMA.md` — table/column reference and storage format.
- `API.md` — live endpoint reference with curl/Python examples.
- `ROADMAP.md` — must-dos (FP4 selection before Blackwell!), scale-up plan, open ideas.
- `CLUSTER-VIABILITY-ANALYSIS.md` — the single consolidated research document:
  DeepSeek V4 Flash 0731 model spec (verified against the official release),
  the used-GPU cluster economics vs verified API pricing, and the research
  history with a 15-entry error ledger. Verdict: PP43 + W4A4 + FP8 activations
  on 43x RTX 3080 10GB = 252K tok/s at ~$0.006/1M all-in — 28x under official
  uncached list pricing, 18x under Sail Research's cached-uncached spread, and
  8.8x under the strictest same-model cached blend; break-even 2-7% duty
  depending on workload cache profile. Reproduce with
  `tools/cluster_model.py` (the .py is the authority). Three earlier research
  reports were merged into it and retired — originals in git history (commit
  3e98f0c); do not resurrect their economics without reading the error ledger
  (section 11) first.

## Sources

- Specs/perf: <https://www.techpowerup.com/gpu-specs> — fetched with headed
  Chromium (`playwright-cli`; plain HTTP gets a PoW challenge, headless gets a
  hard 403 — see `docs/PIPELINE.md`). Raw pages archived hash-verified in the DB.
- Prices: <https://secondhandsilicon.com/> — public JSON API over plain HTTPS
  (`/api/prices/{slug}` + `/api/export/{slug}`), used and new conditions.
