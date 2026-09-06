# Methodology

How "AI TFLOPS per dollar" is defined, and the rules that keep it honest.
Locked decisions live in `summary.decisions` inside `gpu.db` (`runs` table).

## The metric

```
AI_TFLOPS  = dense max(FP4, INT4)      # from TPU Matrix Performance
price      = SHS trailing-30d mean, used eBay sold listings
metric     = AI_TFLOPS / price          # + sparse variant alongside
```

- **Precision selection.** Ampere and Turing have no FP4 hardware, so INT4
  dense is the effective input today. ⚠️ The current code (`run.py`, `ai_t`)
  reads INT4 directly — **before adding any FP4-capable card (Blackwell+),
  this must become `max()` over available precisions** (see `docs/ROADMAP.md`).
- **Sparsity is never primary.** Sparse (2:4) throughput is stored alongside
  (`ai_ts`, `ai_pds`) but ranking uses dense. The multiplier comes from TPU's
  sparse note (Ampere 2×); architectures without it (Turing) get 1×, not 2×.
- **Theoretical fallback.** Where Matrix numbers are absent, the highest
  Theoretical number is used instead. Not yet exercised — all 19 GPUs have
  Matrix blocks; a Tensor-less card (e.g. GTX 16-class) would take this path.
- **Auxiliary metrics** (all stored): VRAM/$, bandwidth/$, W/TFLOP,
  MSRP depreciation + retention in basis points.

## Price rules

- **Canonical price = the SHS API stat** (`stats.average`), because anyone can
  verify it on secondhandsilicon.com. Our independent recompute from raw
  listings is stored as cross-check (typical agreement <1%).
- **Thin-data rule:** 30-day used sample `n < 10` → `price_flag: "thin"`
  (displayed, never silently excluded).
- **Used ranks; new is tracked.** New-condition windows/monthlies are stored
  for context (new-vs-used spreads are themselves analytical signal).
- Coverage, currency, market: USD, eBay US via SHS. No regional handling.

## Spec provenance rules

- Live TPU page > curated baseline; every run diffs live values against
  curated and records `tpu_discrepancies` (empty = verified).
- `status.tpu = "ok"` only for `live_headed` / `live_fetch` / `sibling`
  (TPU-derived). `curated` fallback (no live TPU contact) is `"missing"` —
  check `provenance.tpu_method` for the full story.
- **Sibling policy:** SKUs with no TPU reference page (quiet AIB-only
  launches: RTX 3060 8GB, RTX 2060 12GB) inherit a sibling SKU's reference
  data, with the memory subsystem from TPU's own reporting. Uncertain fields
  (e.g. 2060-12GB MSRP — no official figure exists) are flagged in code
  comments, and `tpu_url` is `null`, never guessed.
- **Relperf policy:** TPU gaming table names are matched exactly (aliases in
  `config.py`). Cards absent from TPU's table stay `null` — a sibling's score
  is never attributed.

## Known limitations

- Single spec source (TPU) + single price source (SHS, eBay US, USD).
- SHS has no slug for some SKUs (e.g. RTX 3050 6GB exists on TPU but is
  unscored here until price coverage exists).
- TPU review-aggregate gaming data (`relperf`) is page-relative (% of that
  page's own card) and only covers reviewed cards — a bonus signal, not a metric.
- MSRP is informational (depreciation only); street reality is the used price.
