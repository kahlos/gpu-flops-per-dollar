-- gpu.db schema v1. Scalars queryable; bulk data (listings, raw pages,
-- history) as zstd-compressed JSON blobs. Money in cents, rates in
-- centi-units (x100), small rates (bw, clocks, density) x10.
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS gpus(
  id TEXT PRIMARY KEY, short_name TEXT, full_name TEXT,
  tpu_page TEXT, tpu_slug TEXT, shs_slug TEXT,
  tpu_method TEXT, tpu_at TEXT, shs_at TEXT, collected_at TEXT,
  status_tpu TEXT, status_pu TEXT, status_pn TEXT,
  relperf_name TEXT, discrepancies TEXT,
  die TEXT, architecture TEXT, process_nm INTEGER, transistors_m INTEGER,
  die_size_mm2 INTEGER, sm INTEGER, cuda INTEGER, tensor INTEGER, rt INTEGER,
  tmu INTEGER, rop INTEGER, base_mhz INTEGER, boost_mhz INTEGER,
  mem_gbps_x10 INTEGER, vram_gb INTEGER, mem_type TEXT, mem_bus INTEGER,
  bw_x10 INTEGER, tdp_w INTEGER, psu_w INTEGER, power TEXT, bus_if TEXT,
  msrp_usd INTEGER, chip TEXT, foundry TEXT, process_detail TEXT, l2 TEXT,
  outputs TEXT, slot TEXT, launch_iso TEXT, tpu_release_iso TEXT,
  tpu_announced_iso TEXT, generation TEXT, predecessor TEXT, successor TEXT,
  production TEXT, driver TEXT, l1 TEXT, density_x10 INTEGER,
  cuda_ver TEXT, directx TEXT, opengl TEXT, opencl TEXT, vulkan TEXT,
  shader TEXT, num_vector TEXT, num_matrix TEXT,
  len_mm INTEGER, hgt_mm INTEGER, wid_mm INTEGER, board_no TEXT,
  form_factor TEXT, interconnect TEXT, mig INTEGER, currency TEXT, market TEXT,
  t_f32 INTEGER, t_f16 INTEGER, t_f64 INTEGER, t_px INTEGER, t_tx INTEGER,
  m_fp4 INTEGER, m_fp8 INTEGER, m_i4 INTEGER, m_i8 INTEGER, m_f16 INTEGER,
  m_bf INTEGER, m_tf INTEGER, m_sparse_bps INTEGER,
  ai_t INTEGER, ai_ts INTEGER, ai_prec TEXT, ai_pd INTEGER, ai_pds INTEGER,
  dep_bps INTEGER, ret_bps INTEGER
);

CREATE TABLE IF NOT EXISTS monthly(
  gpu_id TEXT, cond TEXT, mkey TEXT,
  avg_c INTEGER, min_c INTEGER, max_c INTEGER, std_c INTEGER,
  tot INTEGER, new_n INTEGER, used_n INTEGER, chg_bps INTEGER,
  PRIMARY KEY (gpu_id, cond, mkey));

CREATE TABLE IF NOT EXISTS windows(
  gpu_id TEXT, cond TEXT, w TEXT,
  n INTEGER, avg_c INTEGER, med_c INTEGER, lo_c INTEGER, hi_c INTEGER,
  PRIMARY KEY (gpu_id, cond, w));

CREATE TABLE IF NOT EXISTS cond_stats(
  gpu_id TEXT, cond TEXT,
  api_avg_c INTEGER, cur_c INTEGER, lo_c INTEGER, hi_c INTEGER, cnt INTEGER,
  chg_bps INTEGER, yr_hi_c INTEGER, yr_lo_c INTEGER, tot INTEGER,
  bench INTEGER, brand TEXT, series TEXT, base_price INTEGER,
  canon_c INTEGER, fetched_at TEXT,
  PRIMARY KEY (gpu_id, cond));

CREATE TABLE IF NOT EXISTS listing_blobs(
  gpu_id TEXT, cond TEXT, d0 TEXT, n_rows INTEGER, n_titles INTEGER,
  data BLOB, sha TEXT, PRIMARY KEY (gpu_id, cond));

CREATE TABLE IF NOT EXISTS relperf(
  name TEXT PRIMARY KEY, pct INTEGER);

CREATE TABLE IF NOT EXISTS history(
  ts TEXT, gpu_id TEXT, data BLOB, sha TEXT, kind TEXT, base_ts TEXT,
  PRIMARY KEY (ts, gpu_id));

CREATE TABLE IF NOT EXISTS runs(
  ts TEXT PRIMARY KEY, summary TEXT, manifest TEXT);

CREATE TABLE IF NOT EXISTS raw_pages(
  gpu_id TEXT PRIMARY KEY, html BLOB, sha TEXT, fetched_at TEXT);
