/* Renders live data from GET /api/bundle (see tools/serve.py).
   Live-only: start the server first (./serve), then open /website/. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const board = $("#board"), tbody = $("#tbody"), cards = $("#cards");
  const sortSel = $("#sortSel"), q = $("#q"), sparseToggle = $("#sparseToggle");

  let DATA = [];
  let SUMMARY = null;

  async function ensureData() {
    // Live-only: the single source of truth is gpu.db behind /api/bundle.
    // Absolute path: this page is served from /website/ but the API is at /api/*.
    const r = await fetch("/api/bundle", { cache: "no-store" });
    if (!r.ok) throw new Error("api/bundle returned " + r.status);
    const live = window.expandBundle(await r.json());
    DATA = live.data;
    SUMMARY = live.summary;
  }

  const val = (r) => sparseToggle.checked ? r.ai_compute.ai_tflops_per_dollar_sparse : r.ai_compute.ai_tflops_per_dollar;
  const tflops = (r) => sparseToggle.checked ? r.ai_compute.ai_tflops_sparse : r.ai_compute.ai_tflops;

  function filtered() {
    const needle = (q.value || "").toLowerCase().trim();
    let rows = DATA.filter((r) =>
      !needle || (r.short_name + " " + r.specs.memory_size_gb + "GB " + r.specs.memory_type).toLowerCase().includes(needle));
    const k = sortSel.value;
    const num = (r) => ({
      tflops_per_dollar: r.ai_compute.ai_tflops_per_dollar,
      tflops_per_dollar_sparse: r.ai_compute.ai_tflops_per_dollar_sparse,
      ai_tflops: r.ai_compute.ai_tflops,
      price_usd: r.pricing ? r.pricing.price_usd_last30d_avg_api : 1e18,
      memory_gb: r.specs.memory_size_gb, bandwidth_gbs: r.specs.memory_bandwidth_gbs,
      tdp_w: r.specs.tdp_w, short_name: 0,
    }[k]);
    rows.sort((a, b) => k === "price_usd" || k === "tdp_w" ? num(a) - num(b)
      : k === "short_name" ? a.short_name.localeCompare(b.short_name) : num(b) - num(a));
    return rows;
  }

  function sparkline(buckets) {
    if (!buckets || !buckets.length) return "";
    const pts = buckets.slice(-24);
    const w = 300, h = 52, p = 4;
    const vs = pts.map((b) => b.avgPrice);
    const mn = Math.min(...vs), mx = Math.max(...vs), rg = (mx - mn) || 1;
    const d = pts.map((b, i) =>
      `${(p + (i * (w - 2 * p)) / (pts.length - 1)).toFixed(1)},${(h - p - ((b.avgPrice - mn) / rg) * (h - 2 * p)).toFixed(1)}`).join(" ");
    return `<svg class="spark" viewBox="0 0 ${w} ${h}"><polyline points="${d}" fill="none" stroke="#5eead4" stroke-width="2"/>` +
      `<text x="${w - p}" y="${p + 9}" text-anchor="end" font-size="10" fill="#93a0b8">$${Math.round(mx)} recent high</text></svg>`;
  }

  function render() {
    const rows = filtered();
    const max = Math.max(...rows.map(val), 1e-9);
    document.querySelectorAll("#tbl th").forEach((th) => {
      th.onclick = () => {
        const map = { short_name: "short_name", ai_tflops: "ai_tflops", price_usd: "price_usd", tflops_per_dollar: "tflops_per_dollar", memory_gb: "memory_gb", bandwidth_gbs: "bandwidth_gbs", tdp_w: "tdp_w", bus_interface: "short_name" };
        sortSel.value = map[th.dataset.k] || "tflops_per_dollar"; render();
      };
    });

    board.innerHTML = rows.map((r, i) => `
      <div class="row${i === 0 ? " top" : ""}">
        <div class="nm">${i === 0 ? "👑 " : ""}${r.short_name}<small>${r.ai_compute.ai_precision_used} · ${r.specs.memory_size_gb}GB ${r.specs.memory_type}</small></div>
        <div class="bar"><i style="width:${(100 * val(r) / max).toFixed(1)}%"></i></div>
        <div class="val"><b>${val(r).toFixed(3)} TFLOPS/$</b><span>${tflops(r).toFixed(1)} TFLOPS · $${r.pricing.price_usd_last30d_avg_api}</span></div>
      </div>`).join("");

    tbody.innerHTML = rows.map((r) => `
      <tr><td><a href="#${r.id}">${r.short_name}</a></td><td class="num">${tflops(r).toFixed(1)}</td>
      <td class="num">$${r.pricing.price_usd_last30d_avg_api}</td><td class="num"><strong>${val(r).toFixed(3)}</strong></td>
      <td class="num">${r.specs.memory_size_gb} GB</td><td class="num">${r.specs.memory_bandwidth_gbs}</td>
      <td class="num">${r.specs.tdp_w} W</td><td>${r.specs.bus_interface}</td></tr>`).join("");

    cards.innerHTML = rows.map((r) => {
      const p = r.pricing, d = p ? p.price_30d_detail : {};
      const q = p.quality || {flag: "ok", n_30d: d.count_30d};
      const thin = q.flag === "thin" ? `<span class="badge thin">⚠ thin price data (n=${q.n_30d})</span>` : "";
      const m = r.matrix_performance;
      return `<article class="card" id="${r.id}">
        <h3>${r.short_name}</h3>
        <div class="badges">
          <span class="badge hot">${val(r).toFixed(3)} TFLOPS/$${sparseToggle.checked ? " sparse" : ""}</span>
          <span class="badge">${r.specs.memory_size_gb}GB ${r.specs.memory_type}</span>
          <span class="badge">${r.specs.memory_bandwidth_gbs} GB/s</span>
          <span class="badge">${r.specs.tdp_w}W · ${r.specs.bus_interface}</span>
          ${thin}
        </div>
        <div class="kv">
          <div><span>AI INT4 dense / sparse</span><b>${r.ai_compute.ai_tflops.toFixed(1)} / ${r.ai_compute.ai_tflops_sparse.toFixed(1)} T</b></div>
          <div><span>INT8 / FP16 tensor</span><b>${m.int8_dense_tops.toFixed(1)} / ${m.fp16_dense_tflops.toFixed(1)} T</b></div>
          <div><span>BF16 / TF32 tensor</span><b>${m.bf16_dense_tflops != null ? m.bf16_dense_tflops.toFixed(2) : "—"} / ${m.tf32_dense_tflops != null ? m.tf32_dense_tflops.toFixed(2) : "—"} T</b></div>
          <div><span>Used 30d avg</span><b>$${p.price_usd_last30d_avg_api} (n=${q.n_30d})</b></div>
          <div><span>30d median/range</span><b>$${p.price_median_30d} · $${d.min_30d}–$${d.max_30d}</b></div>
          <div><span>FP32 / FP16 theo</span><b>${r.theoretical_performance.fp32_tflops} T</b></div>
          <div><span>CUDA / SM / TMU / ROP</span><b>${r.specs.cuda_cores} / ${r.specs.sm_count} / ${r.specs.tmu} / ${r.specs.rop}</b></div>
          <div><span>Boost / mem</span><b>${r.specs.boost_clock_mhz} MHz · ${r.specs.memory_clock_effective_gbps} Gbps</b></div>
          <div><span>VRAM/$ · W/TFLOP</span><b>${r.vram_per_dollar_gb} GB · ${r.watts_per_tflops} W</b></div>
        </div>
        ${sparkline(p.monthly_buckets)}
        <div class="links">${r.sources.tpu_url ? `<a href="${r.sources.tpu_url}">TPU specs ↗</a>` : `<span style="color:var(--mut)" title="No TPU reference page exists; derived from sibling SKU">TPU: sibling-derived</span>`}<a href="${r.sources.shs_url}">SHS price ↗</a>
        <span style="color:var(--mut)">MSRP $${r.specs.launch_msrp_usd} · ${r.specs.launch_date}</span></div>
      </article>`;
    }).join("");

    const best = [...DATA].sort((a, b) => b.ai_compute.ai_tflops_per_dollar - a.ai_compute.ai_tflops_per_dollar)[0];
    const vramKing = [...DATA].sort((a, b) => b.specs.memory_size_gb - a.specs.memory_size_gb || a.pricing.price_usd_last30d_avg_api - b.pricing.price_usd_last30d_avg_api)[0];
    $("#heroStats").innerHTML = `
      <div class="stat"><b>${best.short_name}</b><span>best value · ${best.ai_compute.ai_tflops_per_dollar.toFixed(3)} TFLOPS/$</span></div>
      <div class="stat"><b>${DATA.length} GPUs</b><span>${SUMMARY.coverage || ""} · dense max(FP4,INT4)</span></div>
      <div class="stat"><b>$${Math.min(...DATA.map((r) => r.pricing.price_usd_last30d_avg_api))}–$${Math.max(...DATA.map((r) => r.pricing.price_usd_last30d_avg_api))}</b><span>used 30-day range</span></div>
      <div class="stat"><b>${vramKing.short_name} ${vramKing.specs.memory_size_gb}GB</b><span>VRAM pick · $${vramKing.pricing.price_usd_last30d_avg_api}</span></div>`;
    if (SUMMARY) $("#genAt").textContent =
      "Live from gpu.db · " + new Date(SUMMARY.generated_at).toLocaleString();
    if (SUMMARY && SUMMARY.coverage) {
      document.querySelector("#coverageName").textContent = SUMMARY.coverage;
      document.title = `AI TFLOPS per Dollar — ${SUMMARY.coverage} | GPU Flops per Dollar`;
    }
  }

  sortSel.onchange = render; q.oninput = render; sparseToggle.onchange = render;
  ensureData().then(render).catch((e) => {
    board.innerHTML = `<p style="color:#f87171">Could not load live data (${e.message}). ` +
      `Start the server with <code>./serve</code> and open ` +
      `<code>http://localhost:8000/website/</code> — this page reads only from gpu.db, ` +
      `there is no offline snapshot.</p>`;
  });
})();
