/* Expands compact v2 bundles (from GET /api/bundle) into the render shape. */
(function () {
  "use strict";
  const pool_of = (DB) => DB.pool;
  function expandBundle(DB) {
  const pool = DB.pool;
  const P = (i) => (i === null || i === undefined ? null : pool[i]);
  const PF = ["ok", "thin", "missing"];
  const D = (c) => (c === null || c === undefined ? null : c / 100);

  function cond(c) {
    if (!c) return null;
    const w = {};
    for (const k of Object.keys(c.w)) {
      const r = c.w[k];
      w[k] = { n: r[0], avg: D(r[1]), med: D(r[2]), lo: D(r[3]), hi: D(r[4]) };
    }
    return {
      avg_usd: D(c.avg), api_avg_usd: D(c.api),
      canon_usd: c.api !== null && c.api !== undefined ? D(c.api) : D(c.avg),
      windows: w,
      monthly: c.m.map((r) => ({ mkey: r[0], avg: D(r[1]), min: D(r[2]), max: D(r[3]), std: D(r[4]), tot: r[5], nw: r[6], used: r[7], chg: r[8] / 100 })),
      chg_pct: c.chg / 100, yr_hi: D(c.yh), yr_lo: D(c.yl), total: c.tot, bench: c.bn,
    };
  }

  const data = DB.gpus.map((c) => {
    const s = c.sp, u = cond(c.pr.u);
    const price = u ? u.canon_usd : null;
    const w30 = u ? u.windows["30d"] : { n: 0, avg: null, med: null, lo: null, hi: null };
    const aiT = c.ai.t / 100, aiTs = c.ai.ts / 100;
    const bw = s.bw / 10, vgb = s.vgb;
    return {
      id: c.i, short_name: P(c.n), name: P(c.f),
      sources: {
        tpu_url: c.src.t && c.src.ts !== null ? `https://www.techpowerup.com/gpu-specs/${P(c.src.ts)}.${c.src.t}` : null,
        shs_url: `https://secondhandsilicon.com/product/${c.src.s}`, shs_slug: c.src.s,
        tpu_collection: ["live_headed", "live_fetch", "curated", "sibling", "missing"][c.pv.tm],
      },
      specs: {
        memory_size_gb: vgb, memory_type: P(s.vty), memory_bandwidth_gbs: bw, tdp_w: s.tdp,
        bus_interface: P(s.bif), boost_clock_mhz: s.kcl, memory_clock_effective_gbps: s.mcl / 10,
        cuda_cores: s.cud, sm_count: s.sm, tmu: s.tmu, rop: s.rop,
        launch_msrp_usd: s.msrp, launch_date: s.lch,
      },
      matrix_performance: {
        int4_dense_tops: D(c.mx.i4), int8_dense_tops: D(c.mx.i8),
        fp16_dense_tflops: D(c.mx.mf), bf16_dense_tflops: D(c.mx.bf),
        tf32_dense_tflops: D(c.mx.tf),
      },
      ai_compute: {
        ai_tflops: aiT, ai_tflops_sparse: aiTs,
        ai_tflops_per_dollar: c.ai.pd / 10000, ai_tflops_per_dollar_sparse: c.ai.pds / 10000,
        ai_precision_used: P(c.ai.pu),
      },
      theoretical_performance: { fp32_tflops: c.th.f32 / 100 },
      pricing: u ? {
        price_usd_last30d_avg_api: price,
        price_30d_detail: { count_30d: w30.n, min_30d: w30.lo, max_30d: w30.hi },
        price_median_30d: w30.med,
        monthly_buckets: u.monthly.map((b) => ({ avgPrice: b.avg })),
        quality: { flag: PF[c.st.pu], n_30d: w30.n },
      } : null,
      vram_per_dollar_gb: price ? +(vgb / price).toFixed(4) : null,
      bandwidth_per_dollar: price ? +(bw / price).toFixed(4) : null,
      watts_per_tflops: +(s.tdp / aiT).toFixed(3),
    };
  });
  return { data, summary: DB.summary };
  }
  window.expandBundle = expandBundle;
})();
