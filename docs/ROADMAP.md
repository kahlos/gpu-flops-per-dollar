# Roadmap

Prioritized future work. Status context: 19 GPUs (RTX 20 + 30), single-file
`gpu.db`, live API + site, manual crawls.

## Must-do before new architectures

- **FP4 selection logic.** `run.py` hardcodes `ai_t = matrix INT4` (correct for
  Turing/Ampere, which lack FP4). Before adding any Blackwell+ card, implement
  `max()` over available precisions per `docs/METHODOLOGY.md`, with
  `ai_precision_used` recording the winner. The `fp4`/`fp8` columns and null
  handling already exist.
- **`record_sha` is frozen.** Its canonical form (timestamp-excluded expanded
  doc) defines history identity — changing it re-baselines every GPU's history.
  Treat edits as migrations.

## Scale-up (the stated goal: all historical GPUs)

- **Discovery automation:** script `config.py` growth from SHS `/api/search`
  sweeps + the TPU database table (headed session), emitting candidate entries
  with source URLs for human review.
- **Batch dump tooling:** `fetch_tpu_live.sh` has a hardcoded URL map — generate
  it from `config.py` (missing-page SKUs only) so dumps stay in sync.
- **GTX 16-class cards** exercise the theoretical-fallback path for the first
  time (no Tensor cores) — good canary for methodology edge cases.
- **Workstation/DC cards:** schema fields already reserved (`form_factor`,
  `interconnect`, `mig`); SHS coverage will be thinner — expect `thin` flags.
- **Crawl cadence:** weekly full runs are plenty (specs static, SHS daily);
  monitor SHS call volume (~4 calls/GPU).

## Data quality

- **Outlier analysis on archived titles:** listing titles are kept precisely so
  AIB-variant splits ("FTW3", "for parts", bulk lots) can be studied later.
- **Thin-data revisits:** widen windows or longer accumulation for low-n SKUs
  instead of single-point 30d means.
- **MSRP sourcing:** 2060-12GB-class gaps need a documented second source rule.
- **Second spec source:** cross-check high-leverage fields (clocks, TDP) against
  NVIDIA whitepapers/board data, not just TPU.

## Product (site + API)

- **Price-history charts** on the site from `monthly` + `/api/.../prices`
  (data exists; only dense leaderboard exists today).
- **CSV export endpoint** (`/api/gpus?format=csv`) for spreadsheet agents.
- **Filtering/sorting on `/api/gpus`** (by arch, VRAM, price band).
- **Currency/region support** if scope leaves eBay-US.
- Serve is localhost-only by design; any exposure needs auth + rate limits first.

## Engineering hygiene

- **pytest suite** (currently verification is manual scripts): pack round-trip,
  delta round-trip + chain resolution, API smoke, summary-full-DB invariant.
- **Git + backup policy:** no repo exists yet; `gpu.db` is THE database — decide
  backup cadence before it grows (single file ⇒ trivially copyable; never
  copy it mid-write — checkpoint first or copy with SQLite backup API).
- **Serve hardening:** request logging exists; add `--bind` deliberately (never
  default to `0.0.0.0`), consider read-only filesystem posture.
- **Playwright pinning:** `playwright-cli` binary + headed-Chromium behavior is
  load-bearing and unpinned — note versions when the crawl breaks.
