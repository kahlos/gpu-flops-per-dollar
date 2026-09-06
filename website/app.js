/* GPU Compute-per-Dollar explorer: per-precision rankings, never mixing FP4 with INT4.
   Live-only: reads GET /api/bundle (see tools/serve.py + website/dbloader.js). */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const board = $("#board"), tbody = $("#tbody"), tblHead = $("#tblHead"),
    cards = $("#cards"), scatterSvg = $("#scatter"), scatterTip = $("#scatterTip"),
    heroStats = $("#heroStats"), precGroups = $("#precGroups"), precWarn = $("#precWarn"),
    precSupport = $("#precSupport"), vendorChips = $("#vendorChips"),
    archSel = $("#archSel"), genSel = $("#genSel"), vramSel = $("#vramSel"),
    priceCap = $("#priceCap"), priceCapLbl = $("#priceCapLbl"),
    tdpCap = $("#tdpCap"), tdpCapLbl = $("#tdpCapLbl"),
    q = $("#q"), sortSel = $("#sortSel"),
    hideThin = $("#hideThin"), tensorOnly = $("#tensorOnly"),
    logX = $("#logX"), logY = $("#logY"), paretoToggle = $("#paretoToggle"),
    resultCount = $("#resultCount"), rankSub = $("#rankSub"), rankHint = $("#rankHint"),
    tableSub = $("#tableSub"), cardsSub = $("#cardsSub"), scatterSub = $("#scatterSub"),
    pinnedBar = $("#pinnedBar"), scatterLegend = $("#scatterLegend");

  const PRECISIONS = [
    { key: "fp4", label: "FP4", group: "4-bit", unit: "TFLOPS",
      desc: "4-bit float. Only Blackwell (RTX 50) publishes FP4; excluded elsewhere.",
      get: (r) => r.precisions.fp4 },
    { key: "int4", label: "INT4", group: "4-bit", unit: "TOPS",
      desc: "4-bit integer. Turing→Ada, RDNA4, Arc. Blackwell has no INT4 row.",
      get: (r) => r.precisions.int4 },
    { key: "fp8", label: "FP8", group: "8-bit", unit: "TFLOPS",
      desc: "8-bit float. Ada, Blackwell, RDNA4. Missing on Turing/Ampere/Arc.",
      get: (r) => r.precisions.fp8 },
    { key: "int8", label: "INT8", group: "8-bit", unit: "TOPS",
      desc: "8-bit integer — the most cross-vendor comparable inference format.",
      get: (r) => r.precisions.int8 },
    { key: "fp16", label: "FP16", group: "16-bit", unit: "TFLOPS",
      desc: "Half precision tensor. Nearly universal on tensor silicon.",
      get: (r) => r.precisions.fp16 },
    { key: "bf16", label: "BF16", group: "16-bit", unit: "TFLOPS",
      desc: "Brain float 16. Training-adjacent. Missing on Turing.",
      get: (r) => r.precisions.bf16 },
    { key: "tf32", label: "TF32", group: "32-bit", unit: "TFLOPS",
      desc: "TensorFloat-32. NVIDIA Ampere and newer (no RDNA4/Arc row).",
      get: (r) => r.precisions.tf32 },
    { key: "fp32", label: "FP32*", group: "Fallback", unit: "TFLOPS",
      desc: "Theoretical FP32 (not tensor). Every card has it — the only way to include pre-tensor silicon.",
      get: (r) => r.precisions.fp32_theo },
    { key: "aimax", label: "AI-max*", group: "Fallback", unit: "T",
      desc: "Legacy dense max(FP4,INT4,…). MIXES precisions — FP4 cards rank against INT4 cards. Shown for continuity only.",
      get: (r) => r.ai_compute.ai_tflops },
  ];
  const precByKey = Object.fromEntries(PRECISIONS.map((p) => [p.key, p]));
  const VENDOR_COLORS = { NVIDIA: "#76b900", AMD: "#f87171", Intel: "#60a5fa" };

  let DATA = [], SUMMARY = null;
  const state = {
    prec: "int8", cond: "used", sparse: false, view: "rank",
    vendors: new Set(), arch: "", gen: "", vramMin: 0,
    priceCap: 8200, tdpCap: 600, hideThin: false, tensorOnly: false,
    sort: "perDollar", logX: true, logY: true, pareto: true, pinnedId: null,
  };

  const F1 = (x) => (x == null ? "—" : (+x).toFixed(1));
  const F2 = (x) => (x == null ? "—" : (+x).toFixed(2));
  const F3 = (x) => (x == null ? "—" : (+x).toFixed(3));
  const FM = (x) => (x == null ? "—" : "$" + Math.round(x).toLocaleString());
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function pricingOf(r) { return state.cond === "new" ? r.pricing_new : r.pricing_used; }
  function priceOf(r) { const p = pricingOf(r); return p ? p.price_usd_canon : null; }
  function denseOf(r) {
    if (state.prec === "aimax") return r.ai_compute.ai_tflops;
    return precByKey[state.prec].get(r);
  }
  function tflopsOf(r) {
    const d = denseOf(r);
    if (d == null) return null;
    if (state.prec === "aimax") {
      return state.sparse ? r.ai_compute.ai_tflops_sparse : r.ai_compute.ai_tflops;
    }
    if (state.sparse && r.sparse_mult && r.sparse_mult > 1) return d * r.sparse_mult;
    return d;
  }
  function perDollarOf(r) {
    const t = tflopsOf(r), p = priceOf(r);
    if (t == null || p == null || p <= 0) return null;
    return t / p;
  }
  function wattsPerTflop(r) {
    const t = tflopsOf(r);
    if (t == null || !r.specs.tdp_w) return null;
    return r.specs.tdp_w / t;
  }
  function isThin(r) {
    const p = pricingOf(r);
    return p && p.quality && p.quality.flag === "thin";
  }
  function priceN(r) {
    const p = pricingOf(r);
    return p && p.quality ? p.quality.n_30d : null;
  }
  function isStale(r) { return priceN(r) === 0; }
  function priceBadge(r) {
    const n = priceN(r);
    if (n === 0) return ` <span class="badge thin">stale · no 30d sales</span>`;
    if (isThin(r)) return ` <span class="badge thin">thin n=${n}</span>`;
    return "";
  }

  function supportCount(key) {
    if (key === "aimax") return DATA.filter((r) => r.ai_compute.ai_tflops != null).length;
    if (key === "fp32") return DATA.filter((r) => r.precisions.fp32_theo != null).length;
    const g = precByKey[key].get;
    return DATA.filter((r) => g(r) != null).length;
  }

  function baseFiltered() {
    const needle = (q.value || "").toLowerCase().trim();
    return DATA.filter((r) => {
      if (state.vendors.size && !state.vendors.has(r.vendor)) return false;
      if (state.arch && r.architecture !== state.arch) return false;
      if (state.gen && r.generation !== state.gen) return false;
      if ((r.specs.memory_size_gb || 0) < state.vramMin) return false;
      const p = priceOf(r);
      if (p == null || p > state.priceCap) return false;
      if (r.specs.tdp_w != null && r.specs.tdp_w > state.tdpCap) return false;
      if (state.hideThin && isThin(r)) return false;
      if (state.hideThin && isStale(r)) return false;
      if (state.tensorOnly && !r.hasTensor) return false;
      if (needle) {
        const hay = (r.short_name + " " + r.name + " " + (r.architecture || "") + " " +
          (r.generation || "") + " " + (r.specs.memory_size_gb || "") + "GB " +
          (r.specs.memory_type || "")).toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }
  // Rankable = has selected precision + price. Everything else is "unsupported".
  function rankedRows(rows) {
    const out = [];
    for (const r of rows || baseFiltered()) {
      const t = tflopsOf(r), p = priceOf(r), pd = perDollarOf(r);
      if (t == null || p == null || pd == null) continue;
      out.push(r);
    }
    const k = state.sort;
    const val = (r) => {
      switch (k) {
        case "tflops": return tflopsOf(r) || -1;
        case "priceAsc": return priceOf(r);
        case "priceDesc": return -(priceOf(r) || 0);
        case "vram": return r.specs.memory_size_gb || -1;
        case "bw": return r.specs.memory_bandwidth_gbs || -1;
        case "tdp": return r.specs.tdp_w == null ? 1e18 : r.specs.tdp_w;
        case "eff": { const w = wattsPerTflop(r); return w == null ? 1e18 : w; }
        case "name": return 0;
        default: return perDollarOf(r) || -1;
      }
    };
    out.sort((a, b) => {
      if (k === "name") return a.short_name.localeCompare(b.short_name);
      if (k === "priceAsc" || k === "tdp" || k === "eff") return val(a) - val(b);
      return val(b) - val(a);
    });
    return out;
  }

  function precLabel() {
    const p = precByKey[state.prec];
    let s = p.label + (state.sparse ? " sparse" : " dense");
    if (state.prec === "aimax") s = "AI-max" + (state.sparse ? " sparse" : " dense") + " (mixed)";
    return s;
  }

  // ---------- small SVG helpers ----------
  function sparkline(buckets) {
    if (!buckets || !buckets.length) return "";
    const pts = buckets.slice(-24);
    if (pts.length < 2) return "";
    const w = 300, h = 52, p = 4;
    const vs = pts.map((b) => b.avgPrice).filter((v) => v != null);
    if (!vs.length) return "";
    const mn = Math.min(...vs), mx = Math.max(...vs), rg = (mx - mn) || 1;
    const idx = pts.map((b) => b.avgPrice);
    const d = pts.map((b, i) =>
      `${(p + (i * (w - 2 * p)) / (pts.length - 1)).toFixed(1)},${(h - p - (((b.avgPrice - mn) / rg)) * (h - 2 * p)).toFixed(1)}`).join(" ");
    return `<svg class="spark" viewBox="0 0 ${w} ${h}"><polyline points="${d}" fill="none" stroke="#5eead4" stroke-width="2"/>` +
      `<text x="${w - p}" y="${p + 9}" text-anchor="end" font-size="10" fill="#93a0b8">$${Math.round(mx)} recent high</text></svg>`;
  }

  function historyChart(monthly) {
    if (!monthly || monthly.length < 2) return `<p class="hint">Not enough monthly history.</p>`;
    const W = 640, H = 220, P = { l: 56, r: 12, t: 14, b: 30 };
    const iw = W - P.l - P.r, ih = H - P.t - P.b;
    const avgs = monthly.map((m) => m.avg).filter((v) => v != null);
    if (!avgs.length) return `<p class="hint">No price history.</p>`;
    let mn = Math.min(...monthly.map((m) => m.min != null ? m.min : m.avg)),
        mx = Math.max(...monthly.map((m) => m.max != null ? m.max : m.avg));
    if (!(mx > mn)) { mx = mn * 1.1 + 1; }
    const pad = (mx - mn) * 0.08; mn -= pad; mx += pad;
    const X = (i) => P.l + (i / (monthly.length - 1)) * iw;
    const Y = (v) => P.t + (1 - (v - mn) / (mx - mn)) * ih;
    const band = monthly.map((m, i) => `${X(i).toFixed(1)},${Y(m.max != null ? m.max : m.avg).toFixed(1)}`).join(" ") +
      " " + monthly.map((m, i) => `${X(monthly.length - 1 - i).toFixed(1)},${Y(monthly[monthly.length - 1 - i].min != null ? monthly[monthly.length - 1 - i].min : monthly[monthly.length - 1 - i].avg).toFixed(1)}`).join(" ");
    const line = monthly.map((m, i) => `${X(i).toFixed(1)},${Y(m.avg).toFixed(1)}`).join(" ");
    const ticks = [mn + (mx - mn) * 0.15, mn + (mx - mn) * 0.5, mn + (mx - mn) * 0.85];
    const xlabels = monthly.filter((_, i) => i % Math.ceil(monthly.length / 6) === 0);
    return `<svg class="hist" viewBox="0 0 ${W} ${H}">` +
      ticks.map((t) => `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" stroke="#263252"/><text x="${P.l - 6}" y="${(Y(t) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#93a0b8">$${Math.round(t)}</text>`).join("") +
      `<polygon points="${band}" fill="rgba(94,234,212,.12)"/>` +
      `<polyline points="${line}" fill="none" stroke="#5eead4" stroke-width="2"/>` +
      xlabels.map((m) => { const i = monthly.indexOf(m); return `<text x="${X(i).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#93a0b8">${esc(m.mkey.slice(2))}</text>`; }).join("") +
      `</svg>`;
  }

  // ---------- renderers ----------
  function renderPrecTabs() {
    const groups = {};
    for (const p of PRECISIONS) { (groups[p.group] = groups[p.group] || []).push(p); }
    precGroups.innerHTML = Object.entries(groups).map(([g, ps]) =>
      `<div class="prec-group"><span class="prec-gname">${esc(g)}</span><div class="prec-btns">` +
      ps.map((p) => {
        const n = supportCount(p.key);
        const on = state.prec === p.key ? " on" : "";
        const dim = n === 0 ? " dim" : "";
        return `<button class="prec${on}${dim}" data-prec="${p.key}" role="tab" aria-selected="${state.prec === p.key}" title="${esc(p.desc)}">${esc(p.label)}<small>${n}</small></button>`;
      }).join("") + `</div></div>`).join("");
    const p = precByKey[state.prec];
    const n = supportCount(state.prec);
    precSupport.textContent = `${n} of ${DATA.length} GPUs publish ${p.label.replace("*", "")} · ${p.unit}`;
    precWarn.hidden = false;
    if (state.prec === "aimax") {
      precWarn.className = "prec-warn bad";
      precWarn.innerHTML = `⚠ <strong>AI-max mixes precisions:</strong> Blackwell ranks on FP4 while Ampere/Ada rank on INT4. FP4 ≠ INT4 — switch to INT8 for an apples-to-apples comparison.`;
    } else if (state.prec === "fp4") {
      precWarn.className = "prec-warn info";
      precWarn.innerHTML = `FP4 exists only on Blackwell (RTX 50). ${n} cards rank here — everything else is listed under “lacks FP4 hardware”. Compare Blackwell among itself, or switch to INT8 to compare across vendors.`;
    } else if (state.prec === "int4") {
      precWarn.className = "prec-warn info";
      precWarn.innerHTML = `INT4 has no Blackwell row (TPU lists FP4 instead), so RTX 50 cards are absent here. INT8 covers Blackwell + Ampere + Ada + RDNA4 + Arc in one table.`;
    } else if (state.prec === "fp32") {
      precWarn.className = "prec-warn info";
      precWarn.innerHTML = `FP32 is theoretical throughput, not tensor — the only scale where 150+ cards (including pre-tensor GTX/Radeon) can share one ranking. For AI inference, prefer INT8 / FP16.`;
    } else if (state.prec === "int8") {
      precWarn.className = "prec-warn ok";
      precWarn.innerHTML = `INT8 is published for nearly every tensor card (Turing → Blackwell, RDNA4, Arc) — this is the fairest cross-vendor AI ranking.`;
    } else {
      precWarn.className = "prec-warn info";
      precWarn.innerHTML = esc(p.desc);
    }
  }

  function renderHero(rows) {
    const pd = (r) => perDollarOf(r), tf = (r) => tflopsOf(r), pr = (r) => priceOf(r);
    if (!rows.length) {
      heroStats.innerHTML = `<div class="stat"><b>No matches</b><span>loosen filters</span></div>`;
      return;
    }
    const bestPool = rows.filter((r) => (priceN(r) || 0) > 0);
    const fresh = bestPool.length ? bestPool : rows;
    const best = [...fresh].sort((a, b) => pd(b) - pd(a))[0];
    const bestStale = bestPool.length ? "" : " (stale price)";
    const fast = [...rows].sort((a, b) => tf(b) - tf(a))[0];
    const cheap = [...fresh].sort((a, b) => pr(a) - pr(b))[0];
    const vramKing = [...rows].sort((a, b) => (b.specs.memory_size_gb || 0) - (a.specs.memory_size_gb || 0) || pr(a) - pr(b))[0];
    const vramStale = isStale(vramKing) ? " (stale)" : "";
    heroStats.innerHTML = `
      <div class="stat"><b>${esc(best.short_name)}</b><span>best value · ${F3(pd(best))} ${esc(precByKey[state.prec].label)}/$${bestStale}</span></div>
      <div class="stat"><b>${esc(fast.short_name)}</b><span>fastest · ${F1(tf(fast))} ${esc(precByKey[state.prec].unit)}</span></div>
      <div class="stat"><b>${FM(pr(cheap))}</b><span>cheapest · ${esc(cheap.short_name)}</span></div>
      <div class="stat"><b>${esc(vramKing.short_name)} ${vramKing.specs.memory_size_gb}GB</b><span>VRAM pick · ${FM(pr(vramKing))}${vramStale}</span></div>`;
    $("#gpuCountKick").textContent = `${DATA.length} GPUs`;
  }

  function updateSortOptions() {
    const pl = precByKey[state.prec].label;
    const opts = [
      ["perDollar", `${pl} / $ (dense${state.sparse ? "/sparse" : ""})`],
      ["tflops", `${pl} raw throughput`],
      ["priceAsc", `Used price (low first)`],
      ["priceDesc", `Used price (high first)`],
      ["vram", `VRAM (high first)`],
      ["bw", `Bandwidth (high first)`],
      ["tdp", `TDP (low first)`],
      ["eff", `Watts per ${pl} (low first)`],
      ["name", `Name (A–Z)`],
    ];
    sortSel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("");
    sortSel.value = state.sort;
  }

  function renderCounts(rows, base) {
    const unsupported = base.length - rows.length - base.filter((r) => priceOf(r) == null).length;
    const noPrice = base.filter((r) => priceOf(r) == null).length;
    const lackHw = base.length - rows.length - noPrice;
    resultCount.textContent =
      `${rows.length} ranked on ${precLabel()} (${state.cond})` +
      (lackHw > 0 ? ` · ${lackHw} lack ${precByKey[state.prec].label.replace("*", "")} hardware` : "") +
      (noPrice > 0 ? ` · ${noPrice} over price cap` : "");
    const sub = `${precLabel()} · ${state.cond} 30-day · ${rows.length} GPUs`;
    rankSub.textContent = sub; tableSub.textContent = sub; cardsSub.textContent = sub; scatterSub.textContent = " — " + sub;
    rankHint.textContent = state.prec === "aimax"
      ? "Bar = legacy AI-max per dollar (mixed FP4/INT4 — prefer a single-precision tab)."
      : `Bar = ${precLabel()} per dollar. ${state.sparse ? "Sparse 2:4 applied where supported (×" + "2)." : "Dense throughput."} Missing bars = no hardware for this format.`;
    $("#condName").textContent = state.cond;
    $("#condName2").textContent = state.cond;
  }

  function renderBoard(rows) {
    const max = Math.max(...rows.map((r) => perDollarOf(r) || 0), 1e-9);
    board.innerHTML = rows.map((r, i) => {
      const pd = perDollarOf(r), tf = tflopsOf(r), pr = priceOf(r);
      const staleThin = priceBadge(r);
      const nosparse = (state.sparse && r.sparse_mult <= 1) ? ` <span class="badge">dense (no 2:4)</span>` : "";
      const fallback = !r.hasTensor ? ` <span class="badge">theoretical fallback</span>` : "";
      return `<div class="row${i === 0 ? " top" : ""}" data-id="${esc(r.id)}" tabindex="0" role="button" aria-label="${esc(r.short_name)} detail">
        <div class="nm">${i === 0 ? "👑 " : ""}${esc(r.short_name)}<small>${esc(r.architecture || "")} · ${r.specs.memory_size_gb}GB ${esc(r.specs.memory_type || "")} · ${esc(r.vendor || "")}</small></div>
        <div class="bar"><i style="width:${(100 * pd / max).toFixed(1)}%"></i></div>
        <div class="val"><b>${F3(pd)} ${esc(precByKey[state.prec].label)}/$</b><span>${F1(tf)} ${esc(precByKey[state.prec].unit)} · ${FM(pr)}${staleThin}${nosparse}${fallback}</span></div>
      </div>`;
    }).join("") || `<p class="hint">No GPUs match these filters for ${esc(precLabel())}.</p>`;
    board.querySelectorAll(".row").forEach((el) => {
      el.onclick = () => openModal(el.dataset.id);
      el.onkeydown = (e) => { if (e.key === "Enter") openModal(el.dataset.id); };
    });
  }

  function paretoSet(rows) {
    const byPrice = [...rows].sort((a, b) => priceOf(a) - priceOf(b) || tflopsOf(b) - tflopsOf(a));
    const out = []; let best = -Infinity;
    for (const r of byPrice) {
      const t = tflopsOf(r);
      if (t > best) { out.push(r); best = t; }
    }
    return out;
  }

  function renderScatter(rows) {
    const W = 760, H = 440, M = { l: 62, r: 16, t: 16, b: 52 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    scatterSvg.innerHTML = "";
    scatterLegend.innerHTML = "";
    if (!rows.length) {
      scatterSvg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#93a0b8">No data for these filters</text>`;
      return;
    }
    let prices = rows.map(priceOf), tfl = rows.map(tflopsOf);
    let x0 = Math.min(...prices), x1 = Math.max(...prices),
        y0 = Math.min(...tfl), y1 = Math.max(...tfl);
    if (state.logX) { x0 = Math.max(x0, 1); }
    if (state.logY) { y0 = Math.max(y0, 0.1); }
    const padF = (lo, hi, log) => {
      if (log) { const l0 = Math.log10(lo), l1 = Math.log10(hi); const p = (l1 - l0) * 0.06 || 0.1; return [Math.pow(10, l0 - p), Math.pow(10, l1 + p)]; }
      const p = (hi - lo) * 0.06 || 1; return [Math.max(lo - p, 0), hi + p];
    };
    [x0, x1] = padF(x0, x1, state.logX);
    [y0, y1] = padF(y0, y1, state.logY);
    const X = (v) => state.logX
      ? M.l + ((Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * iw
      : M.l + ((v - x0) / (x1 - x0)) * iw;
    const Y = (v) => state.logY
      ? M.t + (1 - (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * ih
      : M.t + (1 - (v - y0) / (y1 - y0)) * ih;
    const xTicks = niceTicks(x0, x1, state.logX), yTicks = niceTicks(y0, y1, state.logY);
    let s = "";
    for (const t of xTicks) s += `<line x1="${X(t).toFixed(1)}" x2="${X(t).toFixed(1)}" y1="${M.t}" y2="${H - M.b}" stroke="#1e2a4a"/><text x="${X(t).toFixed(1)}" y="${H - M.b + 18}" text-anchor="middle" font-size="11" fill="#93a0b8">$${fmtTick(t)}</text>`;
    for (const t of yTicks) s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" stroke="#1e2a4a"/><text x="${M.l - 8}" y="${(Y(t) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#93a0b8">${fmtTick(t)}</text>`;
    s += `<text x="${M.l + iw / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#93a0b8">price (${state.cond} 30-day mean, ${state.logX ? "log" : "linear"})</text>`;
    s += `<text x="14" y="${M.t + ih / 2}" text-anchor="middle" font-size="12" fill="#93a0b8" transform="rotate(-90 14 ${M.t + ih / 2})">${esc(precLabel())} (${esc(precByKey[state.prec].unit)}, ${state.logY ? "log" : "linear"})</text>`;
    const pareto = state.pareto ? paretoSet(rows) : [];
    const paretoIds = new Set(pareto.map((r) => r.id));
    if (pareto.length > 1) {
      const pts = [...pareto].sort((a, b) => priceOf(a) - priceOf(b))
        .map((r) => `${X(priceOf(r)).toFixed(1)},${Y(tflopsOf(r)).toFixed(1)}`).join(" ");
      s += `<polyline points="${pts}" fill="none" stroke="#fbbf24" stroke-width="1.6" stroke-dasharray="6 4" opacity="0.9"/>`;
    }
    const others = rows.filter((r) => !paretoIds.has(r.id));
    const drawRow = (r, onFrontier) => {
      const cx = X(priceOf(r)), cy = Y(tflopsOf(r));
      const vram = r.specs.memory_size_gb || 8;
      const rad = 3.5 + Math.sqrt(vram) * 1.5;
      const col = VENDOR_COLORS[r.vendor] || "#5eead4";
      const pinned = state.pinnedId === r.id;
      return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${(pinned ? rad + 3 : rad).toFixed(1)}" fill="${col}" ` +
        `fill-opacity="${onFrontier ? 0.95 : 0.55}" stroke="${pinned ? "#fff" : onFrontier ? "#fbbf24" : col}" stroke-width="${pinned ? 2.5 : onFrontier ? 2 : 1}" ` +
        `data-id="${esc(r.id)}" style="cursor:pointer"><title>${esc(r.short_name)} — ${F1(tflopsOf(r))} ${esc(precByKey[state.prec].unit)} · ${FM(priceOf(r))} · ${F3(perDollarOf(r))}/$</title></circle>`;
    };
    s += others.map((r) => drawRow(r, false)).join("");
    s += pareto.map((r) => drawRow(r, true)).join("");
    // Label two landmarks: best value (highest per-$) and fastest (max TFLOPS).
    // They sit at opposite ends of the frontier, so they never collide —
    // the old top-3 cluster (e.g. 2080 Super / 2080 / 2070 Super) overlapped.
    const byPd = [...rows].sort((a, b) => perDollarOf(b) - perDollarOf(a))[0];
    const byTf = [...rows].sort((a, b) => tflopsOf(b) - tflopsOf(a))[0];
    const labelled = [byPd, ...(byTf && byTf.id !== byPd.id ? [byTf] : [])];
    for (const r of labelled) {
      s += `<text x="${(X(priceOf(r)) + 9).toFixed(1)}" y="${(Y(tflopsOf(r)) - 8).toFixed(1)}" font-size="11" fill="#e8edf7" paint-order="stroke" stroke="#0b1120" stroke-width="3">${esc(r.short_name)}</text>`;
    }
    scatterSvg.innerHTML = s;
    scatterLegend.innerHTML =
      Object.entries(VENDOR_COLORS).map(([v, c]) => `<span class="lg"><i style="background:${c}"></i>${esc(v)}</span>`).join("") +
      `<span class="lg"><i class="dash"></i>pareto-optimal</span>` +
      `<span class="lg dim">bubble ∝ VRAM</span>`;
    scatterSvg.querySelectorAll("circle").forEach((c) => {
      c.addEventListener("mousemove", (e) => {
        const r = DATA.find((d) => d.id === c.dataset.id);
        if (!r) return;
        scatterTip.hidden = false;
        scatterTip.innerHTML = `<b>${esc(r.short_name)}</b><span>${F1(tflopsOf(r))} ${esc(precByKey[state.prec].unit)} · ${FM(priceOf(r))} · <b>${F3(perDollarOf(r))}/$</b></span><span class="dim">${esc(r.architecture || "")} · ${r.specs.memory_size_gb}GB · ${r.specs.tdp_w}W</span>`;
        const wrap = scatterSvg.parentElement.getBoundingClientRect();
        scatterTip.style.left = (e.clientX - wrap.left + 14) + "px";
        scatterTip.style.top = (e.clientY - wrap.top - 10) + "px";
      });
      c.addEventListener("mouseleave", () => { scatterTip.hidden = true; });
      c.addEventListener("click", (e) => {
        e.stopPropagation();
        state.pinnedId = state.pinnedId === c.dataset.id ? null : c.dataset.id;
        renderScatter(rows); renderPinned(rows);
      });
    });
    renderPinned(rows);
  }
  function renderPinned(rows) {
    const r = rows.find((d) => d.id === state.pinnedId) || DATA.find((d) => d.id === state.pinnedId);
    if (!r) { pinnedBar.hidden = true; pinnedBar.innerHTML = ""; return; }
    pinnedBar.hidden = false;
    pinnedBar.innerHTML = `<span><b>${esc(r.short_name)}</b> · ${F1(tflopsOf(r))} ${esc(precByKey[state.prec].unit)} · ${FM(priceOf(r))} · <b>${F3(perDollarOf(r))}/$</b> · ${r.specs.memory_size_gb}GB ${esc(r.specs.memory_type || "")}</span>
      <span class="spacer"></span>
      <button id="pinDetail">Full detail</button><button id="pinX">✕</button>`;
    $("#pinDetail").onclick = () => openModal(r.id);
    $("#pinX").onclick = () => { state.pinnedId = null; renderScatter(rows); renderPinned(rows); };
  }

  function niceTicks(lo, hi, isLog) {
    if (isLog) {
      const out = []; let p = Math.pow(10, Math.floor(Math.log10(lo)));
      const top = hi * 1.001;
      while (p < top) {
        for (const m of [1, 2, 5]) { const v = m * p; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
        p *= 10;
        if (out.length > 14) break;
      }
      return out.length ? out : [lo, hi];
    }
    const out = []; const n = 5;
    for (let i = 0; i <= n; i++) out.push(lo + ((hi - lo) * i) / n);
    return out;
  }
  function fmtTick(v) {
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k";
    if (v >= 100) return Math.round(v).toString();
    if (v >= 10) return v.toFixed(v < 20 ? 1 : 0);
    return v.toFixed(1);
  }

  function renderTable(rows) {
    const pl = precByKey[state.prec].label, un = precByKey[state.prec].unit;
    const cols = [
      { k: "short_name", t: "GPU", num: false },
      { k: "tflops", t: `${pl} ${un}`, num: true },
      { k: "price", t: `${state.cond === "new" ? "New $" : "Used $"} (30d)`, num: true },
      { k: "perDollar", t: `${pl}/$`, num: true },
      { k: "vram", t: "VRAM", num: true },
      { k: "bw", t: "BW GB/s", num: true },
      { k: "tdp", t: "TDP", num: true },
      { k: "eff", t: `W/${pl}`, num: true },
      { k: "arch", t: "Arch", num: false },
    ];
    const sortArrow = (k) => {
      const map = { perDollar: "perDollar", tflops: "tflops", price: "priceAsc", vram: "vram", bw: "bw", tdp: "tdp", eff: "eff" };
      if (state.sort === k || state.sort === map[k]) return " ▲";
      return "";
    };
    tblHead.innerHTML = cols.map((c) => `<th data-k="${c.k}" class="${c.num ? "num" : ""}">${esc(c.t)}${sortArrow(c.k)}</th>`).join("");
    tblHead.querySelectorAll("th").forEach((th) => {
      th.onclick = () => {
        const k = th.dataset.k;
        const map = { short_name: "name", tflops: "tflops", price: "priceAsc", perDollar: "perDollar", vram: "vram", bw: "bw", tdp: "tdp", eff: "eff", arch: "name" };
        state.sort = map[k] || "perDollar";
        sortSel.value = state.sort;
        renderActive();
      };
    });
    tbody.innerHTML = rows.map((r) => {
      const thin = priceBadge(r);
      const fb = !r.hasTensor ? ` <span class="badge">theo</span>` : "";
      return `<tr><td><a href="#${esc(r.id)}" data-id="${esc(r.id)}">${esc(r.short_name)}</a>${thin}${fb}</td>` +
        `<td class="num">${F1(tflopsOf(r))}</td><td class="num">${FM(priceOf(r))}</td>` +
        `<td class="num"><strong>${F3(perDollarOf(r))}</strong></td>` +
        `<td class="num">${r.specs.memory_size_gb} GB</td><td class="num">${r.specs.memory_bandwidth_gbs == null ? "—" : r.specs.memory_bandwidth_gbs}</td>` +
        `<td class="num">${r.specs.tdp_w == null ? "—" : r.specs.tdp_w + " W"}</td>` +
        `<td class="num">${wattsPerTflop(r) == null ? "—" : F2(wattsPerTflop(r))}</td>` +
        `<td>${esc(r.architecture || "—")}</td></tr>`;
    }).join("");
    tbody.querySelectorAll("a").forEach((a) => {
      a.onclick = (e) => { e.preventDefault(); openModal(a.dataset.id); };
    });
  }

  function precCell(v, sel) {
    if (v == null) return `<span class="na">—</span>`;
    return sel ? `<b class="sel">${F1(v)}</b>` : F1(v);
  }
  function renderCards(rows) {
    cards.innerHTML = rows.map((r) => {
      const p = pricingOf(r);
      const thin = isStale(r) ? `<span class="badge thin">⚠ stale · no 30d sales</span>`
        : isThin(r) ? `<span class="badge thin">⚠ thin (n=${p.quality.n_30d})</span>` : "";
      const fb = !r.hasTensor ? `<span class="badge">theoretical fallback</span>` : "";
      const sp = r.sparse_mult > 1 ? `<span class="badge">2:4 sparse ×2</span>` : `<span class="badge">no 2:4</span>`;
      const m = r.matrix_performance;
      const sel = state.prec;
      const row4 = (k, lbl, v) => `<div class="${sel === k ? "hl" : ""}"><span>${lbl}</span><b>${v == null ? "—" : F1(v) + " T"}</b></div>`;
      return `<article class="card" id="card-${esc(r.id)}">
        <h3><a href="#${esc(r.id)}" data-id="${esc(r.id)}">${esc(r.short_name)}</a></h3>
        <div class="badges">
          <span class="badge hot">${F3(perDollarOf(r))} ${esc(precByKey[sel].label)}/$</span>
          <span class="badge">${esc(r.vendor || "")} · ${esc(r.architecture || "")}</span>
          <span class="badge">${r.specs.memory_size_gb}GB ${esc(r.specs.memory_type || "")}</span>
          ${thin}${fb}${sp}
        </div>
        <div class="kv prec-grid">
          ${row4("fp4", "FP4", m.fp4_dense)}${row4("int4", "INT4", m.int4_dense_tops)}
          ${row4("fp8", "FP8", m.fp8_dense)}${row4("int8", "INT8", m.int8_dense_tops)}
          ${row4("fp16", "FP16", m.fp16_dense_tflops)}${row4("bf16", "BF16", m.bf16_dense_tflops)}
          ${row4("tf32", "TF32", m.tf32_dense_tflops)}${row4("fp32", "FP32*", r.precisions.fp32_theo)}
        </div>
        <div class="kv">
          <div><span>${esc(precLabel())}</span><b>${F1(tflopsOf(r))} T · ${FM(priceOf(r))}</b></div>
          <div><span>VRAM / BW</span><b>${r.specs.memory_size_gb}GB · ${r.specs.memory_bandwidth_gbs == null ? "—" : r.specs.memory_bandwidth_gbs + " GB/s"}</b></div>
          <div><span>TDP / W-per-T</span><b>${r.specs.tdp_w == null ? "—" : r.specs.tdp_w + "W"} · ${wattsPerTflop(r) == null ? "—" : F2(wattsPerTflop(r))}</b></div>
          <div><span>${state.cond} med/range</span><b>${FM(p.price_median_30d)} · ${FM(p.price_30d_detail.min_30d)}–${FM(p.price_30d_detail.max_30d)}</b></div>
        </div>
        ${sparkline(p.monthly_buckets)}
        <div class="links">${r.sources.tpu_url ? `<a href="${esc(r.sources.tpu_url)}">TPU specs ↗</a>` : `<span class="mut">TPU: sibling-derived</span>`}<a href="${esc(r.sources.shs_url)}">SHS price ↗</a>
        <button class="detail-btn" data-id="${esc(r.id)}">Detail</button></div>
      </article>`;
    }).join("");
    cards.querySelectorAll("a[data-id]").forEach((a) => {
      a.onclick = (e) => { e.preventDefault(); openModal(a.dataset.id); };
    });
    cards.querySelectorAll(".detail-btn").forEach((b) => { b.onclick = () => openModal(b.dataset.id); });
  }

  // ---------- modal ----------
  const modalBackdrop = $("#modalBackdrop"), modalBody = $("#modalBody");
  function openModal(id) {
    const r = DATA.find((d) => d.id === id);
    if (!r) return;
    if (location.hash !== "#" + id) history.replaceState(null, "", "#" + id);
    const pu = r.pricing_used, pn = r.pricing_new;
    const m = r.matrix_performance;
    const allPrec = [
      ["FP4", m.fp4_dense, "Blackwell-only"], ["INT4", m.int4_dense_tops, "excl. Blackwell"],
      ["FP8", m.fp8_dense, ""], ["INT8", m.int8_dense_tops, "most comparable"],
      ["FP16", m.fp16_dense_tflops, ""], ["BF16", m.bf16_dense_tflops, ""],
      ["TF32", m.tf32_dense_tflops, "NVIDIA-only"], ["FP32*", r.precisions.fp32_theo, "theoretical"],
    ];
    const selBlock = pricingOf(r);
    const precRows = allPrec.map(([lbl, v, note]) => {
      if (v == null) return `<tr class="na"><td>${lbl}</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td>${esc(note)}</td></tr>`;
      const sparse = r.sparse_mult > 1 ? (v * r.sparse_mult) : null;
      const pd = selBlock && selBlock.price_usd_canon ? v / selBlock.price_usd_canon : null;
      const pds = sparse && selBlock && selBlock.price_usd_canon ? sparse / selBlock.price_usd_canon : null;
      const isSel = precByKey[state.prec] && precByKey[state.prec].label.replace("*", "") === lbl.replace("*", "");
      return `<tr class="${isSel ? "hl" : ""}"><td><b>${lbl}</b></td><td class="num">${F1(v)}</td>` +
        `<td class="num">${sparse ? F1(sparse) : "—"}</td><td class="num">${pd ? F3(pd) : "—"}${pds ? " / " + F3(pds) : ""}</td><td>${esc(note)}</td></tr>`;
    }).join("");
    const s = r.specs;
    modalBody.innerHTML = `
      <p class="kicker">${esc(r.vendor || "")} · ${esc(r.architecture || "")} · ${esc(r.generation || "")}</p>
      <h2>${esc(r.short_name)} <small class="mut">${esc(r.name)}</small></h2>
      <div class="badges">
        <span class="badge hot">${F3(perDollarOf(r))} ${esc(precByKey[state.prec].label)}/$ (${esc(state.cond)})</span>
        <span class="badge">${s.memory_size_gb}GB ${esc(s.memory_type || "")}</span>
        <span class="badge">${s.memory_bandwidth_gbs == null ? "—" : s.memory_bandwidth_gbs + " GB/s"}</span>
        <span class="badge">${s.tdp_w == null ? "—" : s.tdp_w + "W"}</span>
        ${r.hasTensor ? "" : `<span class="badge">theoretical fallback</span>`}
        ${r.sparse_mult > 1 ? `<span class="badge">sparse ×2</span>` : ""}
      </div>
      <div class="modal-grid">
        <div>
          <h3>All precisions <small>(dense / sparse 2:4 · per-$ on ${esc(state.cond)})</small></h3>
          <table class="mini"><thead><tr><th>Format</th><th class="num">Dense T</th><th class="num">Sparse T</th><th class="num">T/$ dense${r.sparse_mult > 1 ? "/sparse" : ""}</th><th>Note</th></tr></thead>
          <tbody>${precRows}</tbody></table>
          <p class="hint">FP4 and INT4 are different arithmetic — never summed or averaged. AI-max (${esc(r.ai_compute.ai_precision_used || "—")}: ${F1(r.ai_compute.ai_tflops)} T) is shown for continuity only.</p>
          <h3>Price history — ${esc(state.cond)} <small>monthly mean ± min/max</small></h3>
          ${historyChart(selBlock ? selBlock.monthly_full.map((b) => ({ mkey: b.mkey, avg: b.avg, min: b.min, max: b.max })) : [])}
          <div class="kv">
            <div><span>30d mean / median</span><b>${selBlock ? FM(selBlock.price_usd_canon) + " / " + FM(selBlock.price_median_30d) : "—"}</b></div>
            <div><span>30d range (n=${selBlock ? selBlock.price_30d_detail.count_30d : "—"})</span><b>${selBlock ? FM(selBlock.price_30d_detail.min_30d) + "–" + FM(selBlock.price_30d_detail.max_30d) : "—"}</b></div>
            <div><span>Used 30d</span><b>${pu ? FM(pu.price_usd_canon) + " (n=" + pu.price_30d_detail.count_30d + ")" : "—"}</b></div>
            <div><span>New 30d</span><b>${pn ? FM(pn.price_usd_canon) + " (n=" + pn.price_30d_detail.count_30d + ")" : "—"}</b></div>
          </div>
        </div>
        <div>
          <h3>Specs</h3>
          <div class="kv stacked">
            <div><span>Architecture / gen</span><b>${esc(r.architecture || "—")} / ${esc(r.generation || "—")}</b></div>
            <div><span>Process / die</span><b>${s.process_nm == null ? "—" : s.process_nm + "nm"} · ${esc(s.die || "—")}</b></div>
            <div><span>Transistors / size</span><b>${s.transistors_m == null ? "—" : (s.transistors_m / 1000).toFixed(1) + "B"} · ${s.die_size_mm2 == null ? "—" : s.die_size_mm2 + "mm²"}</b></div>
            <div><span>CUDA / SM / T / RT</span><b>${s.cuda_cores == null ? "—" : s.cuda_cores} / ${s.sm_count == null ? "—" : s.sm_count} / ${s.tensor_cores == null ? "—" : s.tensor_cores} / ${s.rt_cores == null ? "—" : s.rt_cores}</b></div>
            <div><span>TMU / ROP</span><b>${s.tmu == null ? "—" : s.tmu} / ${s.rop == null ? "—" : s.rop}</b></div>
            <div><span>Boost / base</span><b>${s.boost_clock_mhz == null ? "—" : s.boost_clock_mhz + " MHz"} / ${s.base_clock_mhz == null ? "—" : s.base_clock_mhz + " MHz"}</b></div>
            <div><span>Memory</span><b>${s.memory_size_gb}GB ${esc(s.memory_type || "")} · ${s.mem_bus_bits == null ? "—" : s.mem_bus_bits + "-bit"} · ${s.memory_clock_effective_gbps == null ? "—" : s.memory_clock_effective_gbps + " Gbps"}</b></div>
            <div><span>Bandwidth</span><b>${s.memory_bandwidth_gbs == null ? "—" : s.memory_bandwidth_gbs + " GB/s"}</b></div>
            <div><span>TDP / PSU</span><b>${s.tdp_w == null ? "—" : s.tdp_w + "W"} / ${s.psu_w == null ? "—" : s.psu_w + "W"}</b></div>
            <div><span>Bus</span><b>${esc(s.bus_interface || "—")}</b></div>
            <div><span>MSRP / launch</span><b>${s.launch_msrp_usd == null ? "—" : "$" + s.launch_msrp_usd} · ${esc(s.launch_date || "—")}</b></div>
            <div><span>Provenance</span><b>${esc(r.sources.tpu_collection)}${r.discrepancies.length ? " · " + r.discrepancies.length + " diffs" : ""}</b></div>
          </div>
          <div class="links">${r.sources.tpu_url ? `<a href="${esc(r.sources.tpu_url)}">TPU specs ↗</a>` : `<span class="mut">TPU: sibling-derived (no reference page)</span>`}<a href="${esc(r.sources.shs_url)}">SHS price ↗</a></div>
          ${sparkline(selBlock ? selBlock.monthly_buckets : [])}
        </div>
      </div>`;
    modalBackdrop.hidden = false;
    document.body.style.overflow = "hidden";
  }
  function closeModal() {
    modalBackdrop.hidden = true;
    document.body.style.overflow = "";
    history.replaceState(null, "", location.pathname + location.search);
  }
  $("#modalX").onclick = closeModal;
  modalBackdrop.addEventListener("click", (e) => { if (e.target === modalBackdrop) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modalBackdrop.hidden) closeModal(); });

  // ---------- panels / view ----------
  function setView(v) {
    state.view = v;
    document.querySelectorAll("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
    $("#rankPanel").hidden = v !== "rank";
    $("#scatterPanel").hidden = v !== "scatter";
    $("#tablePanel").hidden = v !== "table";
    $("#cardsPanel").hidden = v !== "cards";
    renderActive();
  }
  function renderActive() {
    const base = baseFiltered();
    const rows = rankedRows(base);
    renderPrecTabs(); renderHero(rows); renderCounts(rows, base);
    if (state.view === "rank") renderBoard(rows);
    else if (state.view === "scatter") renderScatter(rows);
    else if (state.view === "table") renderTable(rows);
    else renderCards(rows);
    if (SUMMARY) $("#genAt").textContent = "Live from gpu.db · " + new Date(SUMMARY.generated_at).toLocaleString();
  }

  function populateFilters() {
    const vendors = [...new Set(DATA.map((r) => r.vendor).filter(Boolean))].sort();
    vendorChips.innerHTML = vendors.map((v) => {
      const n = DATA.filter((r) => r.vendor === v).length;
      const on = state.vendors.has(v) ? " on" : "";
      return `<button class="chip${on}" data-v="${esc(v)}" aria-pressed="${state.vendors.has(v)}"><i style="background:${VENDOR_COLORS[v] || "#5eead4"}"></i>${esc(v)}<small>${n}</small></button>`;
    }).join("");
    vendorChips.querySelectorAll(".chip").forEach((c) => {
      c.onclick = () => {
        const v = c.dataset.v;
        if (state.vendors.has(v)) state.vendors.delete(v); else state.vendors.add(v);
        c.classList.toggle("on"); c.setAttribute("aria-pressed", state.vendors.has(v));
        renderActive();
      };
    });
    const archs = [...new Set(DATA.map((r) => r.architecture).filter(Boolean))].sort();
    archSel.innerHTML = `<option value="">All archs (${archs.length})</option>` + archs.map((a) => {
      const n = DATA.filter((r) => r.architecture === a).length;
      return `<option value="${esc(a)}">${esc(a)} (${n})</option>`;
    }).join("");
    const gens = [...new Set(DATA.map((r) => r.generation).filter(Boolean))].sort();
    genSel.innerHTML = `<option value="">All generations (${gens.length})</option>` + gens.map((g) => {
      const n = DATA.filter((r) => r.generation === g).length;
      return `<option value="${esc(g)}">${esc(g)} (${n})</option>`;
    }).join("");
    const maxP = Math.max(...DATA.map((r) => priceOf(r) || 0), 100);
    priceCap.max = Math.ceil(maxP / 100) * 100;
    priceCap.value = priceCap.max;
    state.priceCap = +priceCap.max;
    priceCapLbl.textContent = (+priceCap.max).toLocaleString();
    const maxT = Math.max(...DATA.map((r) => r.specs.tdp_w || 0), 100);
    tdpCap.max = Math.ceil(maxT / 25) * 25;
    tdpCap.value = tdpCap.max;
    state.tdpCap = +tdpCap.max;
    tdpCapLbl.textContent = "any";
  }

  function bind() {
    precGroups.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-prec]");
      if (!b) return;
      state.prec = b.dataset.prec;
      state.sort = "perDollar";
      updateSortOptions();
      renderActive();
    });
    document.querySelectorAll("#condSeg button").forEach((b) => {
      b.onclick = () => {
        state.cond = b.dataset.cond;
        document.querySelectorAll("#condSeg button").forEach((x) => x.classList.toggle("on", x === b));
        renderActive();
      };
    });
    document.querySelectorAll("#sparseSeg button").forEach((b) => {
      b.onclick = () => {
        state.sparse = b.dataset.sp === "sparse";
        document.querySelectorAll("#sparseSeg button").forEach((x) => x.classList.toggle("on", x === b));
        updateSortOptions(); renderActive();
      };
    });
    document.querySelectorAll("#viewSeg button").forEach((b) => { b.onclick = () => setView(b.dataset.view); });
    let deb = null;
    q.oninput = () => { clearTimeout(deb); deb = setTimeout(renderActive, 120); };
    sortSel.onchange = () => { state.sort = sortSel.value; renderActive(); };
    archSel.onchange = () => { state.arch = archSel.value; renderActive(); };
    genSel.onchange = () => { state.gen = genSel.value; renderActive(); };
    vramSel.onchange = () => { state.vramMin = +vramSel.value; renderActive(); };
    priceCap.oninput = () => { state.priceCap = +priceCap.value; priceCapLbl.textContent = (+priceCap.value).toLocaleString(); renderActive(); };
    tdpCap.oninput = () => {
      state.tdpCap = +tdpCap.value;
      tdpCapLbl.textContent = (+tdpCap.value >= +tdpCap.max) ? "any" : tdpCap.value + "W";
      renderActive();
    };
    hideThin.onchange = () => { state.hideThin = hideThin.checked; renderActive(); };
    tensorOnly.onchange = () => { state.tensorOnly = tensorOnly.checked; renderActive(); };
    logX.onchange = () => { state.logX = logX.checked; renderActive(); };
    logY.onchange = () => { state.logY = logY.checked; renderActive(); };
    paretoToggle.onchange = () => { state.pareto = paretoToggle.checked; renderActive(); };
    $("#resetBtn").onclick = () => {
      state.vendors.clear(); state.arch = ""; state.gen = ""; state.vramMin = 0;
      state.hideThin = false; state.tensorOnly = false; state.sort = "perDollar";
      q.value = ""; archSel.value = ""; genSel.value = ""; vramSel.value = "0";
      hideThin.checked = false; tensorOnly.checked = false;
      priceCap.value = priceCap.max; state.priceCap = +priceCap.max;
      priceCapLbl.textContent = (+priceCap.max).toLocaleString();
      tdpCap.value = tdpCap.max; state.tdpCap = +tdpCap.max; tdpCapLbl.textContent = "any";
      vendorChips.querySelectorAll(".chip").forEach((c) => { c.classList.remove("on"); c.setAttribute("aria-pressed", "false"); });
      updateSortOptions(); renderActive();
    };
  }

  async function ensureData() {
    const r = await fetch("/api/bundle", { cache: "no-store" });
    if (!r.ok) throw new Error("api/bundle returned " + r.status);
    const live = window.expandBundle(await r.json());
    DATA = live.data;
    SUMMARY = live.summary;
  }

  updateSortOptions(); bind();
  ensureData().then(() => {
    populateFilters(); updateSortOptions();
    // deep link: #rtx-3080-10gb opens detail
    renderActive();
    setView("rank");
    if (location.hash.length > 1) {
      const id = location.hash.slice(1);
      if (DATA.some((r) => r.id === id)) openModal(id);
    }
    if (SUMMARY && SUMMARY.coverage) document.title = `GPU Compute per Dollar — ${DATA.length} GPUs, by precision`;
  }).catch((e) => {
    board.innerHTML = `<p style="color:#f87171">Could not load live data (${esc(e.message)}). ` +
      `Start the server with <code>./serve</code> and open ` +
      `<code>http://localhost:8000/website/</code> — this page reads only from gpu.db.</p>`;
  });
})();
