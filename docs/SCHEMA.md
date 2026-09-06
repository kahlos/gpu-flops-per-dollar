# `gpu.db` schema (v1)

Single SQLite file. Money in **cents**, tensor rates in **centi-units** (×100),
bandwidth/clocks/density in **×10 ints**, percent deltas in **basis points**.
Bulk payloads are **zstd-compressed JSON blobs**. See `dbtools/schema.sql`
for the authoritative DDL.

## Tables

| Table | Rows | Contents |
|---|---|---|
| `meta` | ~4 | `schema`, `last_run`, `relperf_base`, `relperf_note` |
| `gpus` | 1/GPU | identity, provenance, all specs, theoretical, matrix, AI metric, depreciation |
| `monthly` | GPU×cond×month | buckets: avg/min/max/std (¢), totals, new-vs-used counts, change (bps) |
| `windows` | GPU×cond×4 | precomputed 7/30/90/365d stats: n/avg/med/min/max (¢) |
| `cond_stats` | GPU×cond | SHS API stats: api avg, current/lo/hi (¢), counts, 30d change (bps), 1-yr hi/lo, PassMark, brand/series/MSRP, **canonical avg** (API stat, else recompute), fetched_at |
| `listing_blobs` | GPU×cond | zstd `{d0, rows: [[day_offset, cents, title_idx]…], t: [titles]}` — the full raw sales history |
| `relperf` | 174 | Shared TPU gaming relative-performance table (`name → pct`, page's own card implicit @100) |
| `history` | grows on change | zstd snapshots, **only for changed GPUs**: full record first, then op-deltas vs previous (`kind`/`base_ts`; fresh full every 20). Run timestamps excluded from change hashes, so identical reruns write nothing |
| `runs` | 1/run | `ts`, leaderboard `summary` JSON, `manifest` {gpu_id: record_sha} |
| `raw_pages` | GPUs w/ TPU page | zstd full TPU HTML + sha + fetched_at (audit trail; sibling-derived SKUs have none) |

## Key columns (`gpus`)

Identity/provenance: `id, short_name, full_name, tpu_page, tpu_slug, shs_slug,
tpu_method (live_headed|live_fetch|curated|sibling|missing), tpu_at, shs_at,
collected_at, status_tpu|pu|pn (ok|thin|missing), relperf_name, discrepancies`.

Specs: `die, architecture, process_nm, transistors_m, die_size_mm2, sm, cuda,
tensor, rt, tmu, rop, base_mhz, boost_mhz, mem_gbps_x10, vram_gb, mem_type,
mem_bus, bw_x10, tdp_w, psu_w, power, bus_if, msrp_usd, chip, foundry,
process_detail, l2, outputs, slot, launch_iso, tpu_release_iso,
tpu_announced_iso, generation, predecessor, successor, production, driver, l1,
density_x10, cuda_ver, directx, opengl, opencl, vulkan, shader, num_vector,
num_matrix, len_mm, hgt_mm, wid_mm, board_no, form_factor, interconnect, mig,
currency, market`.

Perf: `t_f32, t_f16 (¢T), t_f64 (GFLOPS), t_px, t_tx (×10)`; `m_fp4, m_fp8, m_i4,
m_i8, m_f16, m_bf, m_tf (¢T, NULL when unsupported), m_sparse_bps`;
AI: `ai_t, ai_ts (¢T), ai_prec, ai_pd, ai_pds (per-$ ×10000)`;
`dep_bps, ret_bps` (MSRP depreciation / retention).

## Code map

- Write: `scraper/run.py` → `dbtools/store.py` (`connect, upsert_gpu,
  replace_pricing, put_raw_page, put_relperf, commit_run`).
- Read: `dbtools/query.py` (`open_ro, get_gpu_full, leaderboard, get_listings,
  get_relperf, get_summary, get_runs`). `get_gpu_full` returns the canonical
  expanded shape (same dict `run.py` builds), so analysis code is stable.
- Site: `GET /api/bundle` → pooled compact bundle, expanded in-browser by
  `website/dbloader.js` (the page is live-only; `app.js` is storage-agnostic).
- Packing scheme details (pool/enums/scales): `dbtools/schema.py`,
  `dbtools/pack.py`.
