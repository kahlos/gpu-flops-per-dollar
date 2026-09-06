/* Expands compact v2 bundles (from GET /api/bundle) into the render shape.
   Full-fidelity: exposes every tensor precision separately (FP4 vs INT4 are
   NOT interchangeable), both used+new pricing, and arch/generation/vendor
   for filtering. */
(function () {
  "use strict";
  const pool_of = (DB) => DB.pool;
  function expandBundle(DB) {
  const pool = DB.pool;
  const P = (i) => (i === null || i === undefined ? null : pool[i]);
  const PF = ["ok", "thin", "missing"];
  const D = (c) => (c === null || c === undefined ? null : c / 100);
  const N = (x, f) => (x === null || x === undefined ? null : x / f);

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
      monthly: c.m.map((r) => ({ mkey: r[0], avg: D(r[1]), min: D(r[2]), max: D(r[3]), std: D(r[4]), tot: r[5], nw: r[6], used: r[7], chg: r[8] == null ? null : r[8] / 100 })),
      chg_pct: c.chg == null ? null : c.chg / 100, yr_hi: D(c.yh), yr_lo: D(c.yl), total: c.tot, bench: c.bn,
    };
  }

  function condFull(c, flag) {
    const b = cond(c);
    if (!b) return null;
    const w30 = b.windows["30d"] || { n: 0, avg: null, med: null, lo: null, hi: null };
    return {
      price_usd_canon: b.canon_usd,
      price_usd_avg: b.avg_usd,
      price_usd_api: b.api_avg_usd,
      price_30d_detail: { count_30d: w30.n, min_30d: w30.lo, max_30d: w30.hi },
      price_median_30d: w30.med,
      monthly_buckets: b.monthly.map((m) => ({ mkey: m.mkey, avgPrice: m.avg, minPrice: m.min, maxPrice: m.max })),
      monthly_full: b.monthly,
      windows: b.windows,
      quality: { flag: flag || "ok", n_30d: w30.n },
      chg_pct: b.chg_pct, yr_hi: b.yr_hi, yr_lo: b.yr_lo, total: b.total,
      // legacy aliases used by older renderers
      price_usd_last30d_avg_api: b.canon_usd,
    };
  }

  const data = DB.gpus.map((c) => {
    const s = c.sp, u = cond(c.pr.u), un = cond(c.pr.n);
    const price = u ? u.canon_usd : null;
    const priceNew = un ? un.canon_usd : null;
    const w30 = u ? u.windows["30d"] : { n: 0, avg: null, med: null, lo: null, hi: null };
    const w30n = un ? un.windows["30d"] : { n: 0, avg: null, med: null, lo: null, hi: null };
    const aiT = N(c.ai.t, 100), aiTs = N(c.ai.ts, 100);
    const bw = N(s.bw, 10), vgb = s.vgb;
    const fullName = P(c.f), shortName = P(c.n);
    const vendor = fullName ? fullName.split(" ")[0] : null;
    const arch = P(s.arc), gen = P(s.gen);
    const fp4 = D(c.mx.fp4), fp8 = D(c.mx.fp8), i4 = D(c.mx.i4), i8 = D(c.mx.i8);
    const f16m = D(c.mx.mf), bf = D(c.mx.bf), tf = D(c.mx.tf);
    const spb = c.mx.spb == null ? null : c.mx.spb / 10000; // fraction over 1, e.g. 1.0 = +100%
    const sparseMult = spb == null ? 1.0 : 1.0 + spb;
    const f32t = N(c.th.f32, 100), f16t = N(c.th.f16, 100);
    const precisions = {
      fp4, fp8, int4: i4, int8: i8, fp16: f16m, bf16: bf, tf32: tf,
      fp32_theo: f32t, fp16_theo: f16t,
    };
    const hasTensor = fp4 != null || fp8 != null || i4 != null || i8 != null ||
      f16m != null || bf != null || tf != null;
    const puUsed = cond(c.pr.u), pnUsed = cond(c.pr.n);
    const pricingUsed = puUsed ? condFull(c.pr.u, PF[c.st.pu]) : null;
    const pricingNew = pnUsed ? condFull(c.pr.n, PF[c.st.pn]) : null;
    // legacy `pricing` = used, for backward compat
    const legacyPricing = pricingUsed ? {
      price_usd_last30d_avg_api: pricingUsed.price_usd_canon,
      price_30d_detail: pricingUsed.price_30d_detail,
      price_median_30d: pricingUsed.price_median_30d,
      monthly_buckets: pricingUsed.monthly_buckets,
      quality: pricingUsed.quality,
    } : null;
    return {
      id: c.i, short_name: shortName, name: fullName, vendor,
      architecture: arch, generation: gen,
      sources: {
        tpu_url: c.src.t && c.src.ts !== null ? `https://www.techpowerup.com/gpu-specs/${P(c.src.ts)}.${c.src.t}` : null,
        shs_url: `https://secondhandsilicon.com/product/${c.src.s}`, shs_slug: c.src.s,
        tpu_collection: ["live_headed", "live_fetch", "curated", "sibling", "missing"][c.pv.tm],
      },
      specs: {
        memory_size_gb: vgb, memory_type: P(s.vty), memory_bandwidth_gbs: bw, tdp_w: s.tdp,
        bus_interface: P(s.bif), boost_clock_mhz: s.kcl, base_clock_mhz: s.bcl,
        memory_clock_effective_gbps: N(s.mcl, 10),
        cuda_cores: s.cud, sm_count: s.sm, tensor_cores: s.ten, rt_cores: s.rtc,
        tmu: s.tmu, rop: s.rop,
        launch_msrp_usd: s.msrp, launch_date: s.lch,
        architecture: arch, generation: gen, process_nm: s.prm,
        die: P(s.die), transistors_m: s.trs, die_size_mm2: s.dsz,
        mem_bus_bits: s.bus, psu_w: s.psu, power_connectors: P(s.pwr),
        chip: P(s.chp), foundry: P(s.fdy),
      },
      matrix_performance: {
        fp4_dense: fp4, fp8_dense: fp8,
        int4_dense_tops: i4, int8_dense_tops: i8,
        fp16_dense_tflops: f16m, bf16_dense_tflops: bf,
        tf32_dense_tflops: tf, sparse_mult: sparseMult,
      },
      precisions, hasTensor, sparse_mult: sparseMult,
      ai_compute: {
        ai_tflops: aiT, ai_tflops_sparse: aiTs,
        ai_tflops_per_dollar: N(c.ai.pd, 10000), ai_tflops_per_dollar_sparse: N(c.ai.pds, 10000),
        ai_precision_used: P(c.ai.pu),
      },
      theoretical_performance: { fp32_tflops: f32t, fp16_tflops: f16t, fp64_gflops: c.th.f64 },
      pricing: legacyPricing,
      pricing_used: pricingUsed, pricing_new: pricingNew,
      price_used: price, price_new: priceNew,
      status: { tpu: ["ok", "missing"][c.st.ts], pu: PF[c.st.pu], pn: PF[c.st.pn] },
      provenance: { tpu_method: ["live_headed", "live_fetch", "curated", "sibling", "missing"][c.pv.tm] },
      discrepancies: c.ms || [],
      vram_per_dollar_gb: price ? +(vgb / price).toFixed(4) : null,
      bandwidth_per_dollar: price ? +(bw / price).toFixed(4) : null,
      watts_per_tflops: (aiT ? +(s.tdp / aiT).toFixed(3) : null),
    };
  });
  return { data, summary: DB.summary };
  }
  window.expandBundle = expandBundle;
})();
