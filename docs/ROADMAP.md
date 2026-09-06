# Roadmap

Prioritized future work. Status context: ~150 GPUs (GeForce GTX 400 → RTX
50, Radeon HD 5000 → RX 9000, Arc A/B, Quadro/Tesla, Titans), single-file
`gpu.db`, live API + site, batched crawls (see PIPELINE scale-up lessons).

## Done (bulk expansion)

- **Precision selection.** `run.py` takes the dense `max()` over available
  tensor precisions with `ai.prec` recording the winner (FP4 Blackwell,
  INT4 Ampere/Turing/Ada/RDNA3-4/XMX, FP16-matrix Volta, Theoretical
  fallback for Tensor-less). METHODOLOGY updated.
- **Theoretical-fallback canary.** GTX 16/10-class + GCN/TeraScale cards
  exercise the fallback path (highest Theoretical number).
- **Parser unit traps fixed** (TB/s bandwidth, TFLOPS/GFLOPS Theoretical
  cells, MB memory sizes, million-transistor dice, Mbps HBM clocks).
- **Nullable spec fields** (SM/Tensor/RT/clocks for AMD/Intel/old cards),
  **sparse-capable architecture set**, **vendor-prefix display names**,
  **`tpu_overrides`** for variant/shared pages and dual-GPU board totals.
- **`record_sha` is frozen.** Its canonical form (timestamp-excluded expanded
  doc) defines history identity — changing it re-baselines every GPU's history.
  Treat edits as migrations. (Verified stable across the bulk expansion:
  all pre-existing rows kept their hashes.)

## Scale-up (remaining)

- **Discovery automation:** script `config.py` growth from SHS `/api/search`
  sweeps + the TPU database table (headed session), emitting candidate entries
  with source URLs for human review. (Note: TPU table renders ~101 recent
  rows only; older/workstation IDs come from websearch snippets + headed
  title verification.)
- **Batch dump tooling:** `fetch_tpu_live.sh` has a hardcoded URL map — generate
  it from `config.py` (missing-page SKUs only) so dumps stay in sync.
- **Workstation/DC cards:** schema fields already reserved (`form_factor`,
  `interconnect`, `mig`); SHS coverage will be thinner — expect `thin` flags.
- **Crawl cadence:** weekly full runs are plenty (specs static, SHS daily);
  monitor SHS call volume (~4 calls/GPU).
- **Deferred for lack of TPU reference page** (strict rule; re-verify
  periodically): Quadro K600/K620/K4000/K5000, Tesla K40 (K40c vs K40s
  ambiguous) / M60, GTX 590, HD 6990, Radeon Pro Duo, RTX A2000,
  TITAN X (Pascal, no SHS slug either).
- **Out for lack of SHS price coverage** (re-check `/api/search`
  periodically): RTX 3050 6GB, RTX 5050, Titan V, H100/H200/B200, L40/L4,
  Instinct MI-*, FirePro-*, Flex/Max series, GTX 750 Ti/1050 Ti/1630,
  RX Vega 56/64, R9 280X/285/380-series.

## Data quality

- **Outlier analysis on archived titles:** listing titles are kept precisely so
  AIB-variant splits ("FTW3", "for parts", bulk lots) can be studied later.
- **Thin-data revisits:** widen windows or longer accumulation for low-n SKUs
  instead of single-point 30d means.
- **MSRP sourcing:** 2060-12GB-class gaps need a documented second source rule.
- **Second spec source:** cross-check high-leverage fields (clocks, TDP) against
  NVIDIA whitepapers/board data, not just TPU.

## Product (site + API)

- **Done (2026-09): per-precision explorer** on the site — FP4/INT4/FP8/INT8/
  FP16/BF16/TF32/FP32 rankings that never mix precisions, filters, scatter +
  pareto view (website/app.js). The legacy dense `aimax` ranking remains as a
  continuity fallback only.
- **Price-history charts** on the site from `monthly` + `/api/.../prices`
  (data exists; charts not built yet).
- **CSV export endpoint** (`/api/gpus?format=csv`) for spreadsheet agents.
- **Filtering/sorting on `/api/gpus`** (by arch, VRAM, price band).
- **Currency/region support** if scope leaves eBay-US.
- Serve is localhost-only by design; any exposure needs auth + rate limits first.

## Engineering hygiene

- **pytest suite** (currently verification is manual scripts): pack round-trip,
  delta round-trip + chain resolution, API smoke, summary-full-DB invariant.
  Note `tools/cluster_model.py` already follows the self-checking pattern
  (calibration + floor asserts on every run).
- **Git + backup policy:** repo exists and pushes to GitHub
  (kahlos/gpu-flops-per-dollar); `gpu.db` is THE database and stays gitignored
  by design — still decide a backup cadence before it grows (single file ⇒
  trivially copyable; never copy it mid-write — checkpoint first or copy with
  SQLite backup API).
- **Serve hardening:** request logging exists; add `--bind` deliberately (never
  default to `0.0.0.0`), consider read-only filesystem posture.
- **Playwright pinning:** `playwright-cli` binary + headed-Chromium behavior is
  load-bearing and unpinned — note versions when the crawl breaks.
