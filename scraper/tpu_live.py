"""Live TPU page parser (for HTML dumped via headed Chromium, see tools/fetch_tpu_live.sh).

TPU gpu-specs layout: h2 section heading -> div of dl.clearfix(dt label, dd value).
Returns {section: {label: value}} plus a normalized record matching our schema.
"""

from __future__ import annotations

import re
import datetime as _dt

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_iso_date(text: str | None) -> str | None:
    """'Sep 24th, 2020' -> '2020-09-24'."""
    if not text:
        return None
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d+)\w*,?\s+(\d{4})", text)
    if not m:
        return None
    return f"{m.group(3)}-{_MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"


def parse_mm(text: str | None) -> int | None:
    m = re.search(r"([\d,.]+)\s*mm", (text or "").replace(",", ""))
    return int(float(m.group(1))) if m else None


def parse_relperf(html: str) -> list[tuple[str | None, int | None]]:
    """TPU 'Relative Performance' table: (display name, pct) with the page's own
    card implicit at 100 (its row is omitted)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = []
    for e in soup.find_all("div", class_=re.compile(r"relative-performance-entry$")):
        a = e.find("a")
        num = e.find("div", class_=re.compile("entry__number"))
        nm = clean(a.get_text(" ")) if a else None
        m = re.search(r"(\d+)", num.get_text() if num else "")
        rows.append((nm or None, int(m.group(1)) if m else None))
    return [(n, p) for n, p in rows if n]


def clean(text: str) -> str:
    """Collapse whitespace, including literal backslash-escapes TPU embeds in HTML text."""
    if text is None:
        return ""
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    return " ".join(text.split())


def parse_sections(html: str) -> dict[str, dict[str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: dict[str, dict[str, str]] = {}
    for h in soup.find_all("h2"):
        sec = clean(h.get_text(" "))
        div = h.find_next_sibling()
        if not div or not hasattr(div, "find_all"):
            continue
        pairs: dict[str, str] = {}
        for dl in div.find_all("dl"):
            dt = dl.find("dt")
            dd = dl.find("dd")
            if dt and dd:
                k, v = clean(dt.get_text(" ")), clean(dd.get_text(" "))
                if k and v:
                    pairs[k] = v
        if pairs:
            out[sec] = pairs
    return out


def _f(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"([\d,.]+)", text.replace(",", ""))
    try:
        return float(m.group(1).replace(",", "")) if m else None
    except ValueError:
        return None


def _tflops(text: str | None) -> float | None:
    """Theoretical TFLOPS cell: TPU mixes TFLOPS/GFLOPS units by card age
    (e.g. 4090 FP64 '1.290 TFLOPS' vs 4080 '761.5 GFLOPS'; Pascal FP16 in
    GFLOPS). Normalize to TFLOPS."""
    if not text:
        return None
    v = _f(text)
    if v is None:
        return None
    return v / 1000.0 if "GFLOPS" in text.upper() else v


def _gflops(text: str | None) -> float | None:
    """Theoretical FP64 cell: normalize to GFLOPS (TPU mixes units)."""
    if not text:
        return None
    v = _f(text)
    if v is None:
        return None
    return v * 1000.0 if "TFLOPS" in text.upper() else v


def _transb(text: str | None) -> float | None:
    """Transistors cell: TPU uses billions for big dice, millions for small
    ones (e.g. Oland '950 million'). Normalize to billions."""
    if not text:
        return None
    v = _f(text)
    if v is None:
        return None
    return v / 1000.0 if "MILLION" in text.upper() else v


def _memgb(text: str | None) -> float | None:
    """Memory Size cell: TPU uses GB for modern cards, MB for Fermi-era
    (e.g. GTX 480 '1536 MB'). Normalize to GB."""
    if not text:
        return None
    v = _f(text)
    if v is None:
        return None
    return v / 1024.0 if "MB" in text.upper() and "GB" not in text.upper() else v


def normalize(sections: dict[str, dict[str, str]]) -> dict:
    g = lambda s, k: sections.get(s, {}).get(k)
    gp, gc = sections.get("Graphics Processor", {}), sections.get("Graphics Card", {})
    clk, mem = sections.get("Clock Speeds", {}), sections.get("Memory", {})
    rc, th = sections.get("Render Config", {}), sections.get("Theoretical Performance", {})
    mx, bd = sections.get("Matrix Performance", {}), sections.get("Board Design", {})
    gf, nf = sections.get("Graphics Features", {}), sections.get("Numeric Format Support", {})
    mem_clk = clk.get("Memory Clock", "")
    m_gbps = re.search(r"([\d.]+)\s*Gbps", mem_clk)
    m_mbps = re.search(r"([\d.]+)\s*Mbps", mem_clk)
    if m_gbps:
        mem_eff = float(m_gbps.group(1))
    elif m_mbps:
        mem_eff = float(m_mbps.group(1)) / 1000.0
    else:
        mem_eff = None
    mem_bw_raw = mem.get("Bandwidth", "")
    m_tb = re.search(r"([\d.]+)\s*TB/s", mem_bw_raw)
    bw = float(m_tb.group(1)) * 1000 if m_tb else _f(mem_bw_raw)
    return {
        "chip": gp.get("GPU Name"),
        "die": gp.get("GPU Variant"),
        "architecture": gp.get("Architecture"),
        "foundry": gp.get("Foundry"),
        "process": gp.get("Process Type"),
        "process_nm": _f(gp.get("Process Size")),
        "transistors_b": _transb(gp.get("Transistors")),
        "die_size_mm2": _f(gp.get("Die Size")),
        "release_date": gc.get("Release Date"),
        "launch_price_usd": _f(gc.get("Launch Price")),
        "production": gc.get("Production"),
        "bus_interface": gc.get("Bus Interface"),
        "base_clock_mhz": _f(clk.get("Base Clock")),
        "boost_clock_mhz": _f(clk.get("Boost Clock")),
        "memory_clock_effective_gbps": mem_eff,
        "memory_size_gb": _memgb(mem.get("Memory Size")),
        "memory_type": mem.get("Memory Type"),
        "memory_bus_bits": _f(mem.get("Memory Bus")),
        "memory_bandwidth_gbs": bw,
        "cuda_cores": _f(rc.get("Shading Units")),
        "tmu": _f(rc.get("TMUs")),
        "rop": _f(rc.get("ROPs")),
        "sm_count": _f(rc.get("SM Count")),
        "tensor_cores": _f(rc.get("Tensor Cores")),
        "rt_cores": _f(rc.get("RT Cores")),
        "l2_cache": rc.get("L2 Cache"),
        "pixel_rate_gpixels": _f(th.get("Pixel Rate")),
        "texture_rate_gtexels": _f(th.get("Texture Rate")),
        "theoretical_fp16_tflops": _tflops(th.get("FP16")),
        "theoretical_fp32_tflops": _tflops(th.get("FP32")),
        "theoretical_fp64_gflops": _gflops(th.get("FP64")),
        "matrix_int4_tops": _f((mx.get("INT4") or "")),
        "matrix_int8_tops": _f((mx.get("INT8") or "")),
        "matrix_fp4_tflops": _f((mx.get("FP4") or "")),
        "matrix_fp6_tflops": _f((mx.get("FP6") or "")),
        "matrix_fp8_tflops": _f((mx.get("FP8") or "")),
        "matrix_fp16_tflops": _f((mx.get("FP16") or "")),
        "matrix_bf16_tflops": _f((mx.get("BF16") or "")),
        "matrix_tf32_tflops": _f((mx.get("TF32") or "")),
        "sparse_note": mx.get("Sparse Matrix"),
        "tdp_w": _f(bd.get("TDP")),
        "suggested_psu_w": _f(bd.get("Suggested PSU")),
        "outputs": bd.get("Outputs"),
        "power_connectors": bd.get("Power Connectors"),
        "slot_width": bd.get("Slot Width"),
        "length_mm": parse_mm(bd.get("Length")),
        "height_mm": parse_mm(bd.get("Height")),
        "width_mm": parse_mm(bd.get("Width")),
        "board_number": bd.get("Board Number"),
        "release_iso": parse_iso_date(gc.get("Release Date")),
        "announced_iso": parse_iso_date(gc.get("Announced")),
        "generation": gc.get("Generation"),
        "predecessor": gc.get("Predecessor"),
        "successor": gc.get("Successor"),
        "production": gc.get("Production"),
        "driver_support": gc.get("Driver Support"),
        "l1_cache": rc.get("L1 Cache"),
        "density_m_per_mm2": _f(gp.get("Density")),
        "cuda_version": gf.get("CUDA"),
        "directx": gf.get("DirectX"),
        "opengl": gf.get("OpenGL"),
        "opencl": gf.get("OpenCL"),
        "vulkan": gf.get("Vulkan"),
        "shader_model": gf.get("Shader Model"),
        "numeric_vector": nf.get("Vector"),
        "numeric_matrix": nf.get("Matrix"),
    }
