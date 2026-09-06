"""Pipeline v3: live sources -> gpu.db (SQLite + zstd). The site reads live.

Usage:
    .venv/bin/python -m scraper.run --tpu-live-dir /tmp/tpu_live
    .venv/bin/python -m scraper.run --only rtx-3090 --tpu-live-dir /tmp/tpu_live
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scraper.config import GPUS
from scraper import tpu_curated
from scraper.shs_scraper import fetch_condition
from scraper.tpu_live import normalize as normalize_live, parse_iso_date, parse_sections as parse_live
from scraper.tpu_live import parse_relperf
from scraper.tpu_scraper import fetch_tpu_html
from dbtools.schema import MIN_30D_SAMPLES

DB_PATH = ROOT / "gpu.db"
def db_raw_page(con, gid: str):
    """Specs are static: reuse the raw page stored in the DB (html, fetched_at)."""
    from dbtools.store import zdecomp_bytes
    row = con.execute("SELECT html, fetched_at FROM raw_pages WHERE gpu_id=?", (gid,)).fetchone()
    if not row:
        return None, None
    return zdecomp_bytes(row[0]), row[1]


def load_live(tpu_dir: Path | None, meta: dict, con=None, refresh: bool = False):
    """TPU sourcing order: fresh dump dir > DB copy (specs don't change) >
    live attempt (new GPUs / forced refresh) > curated fallback."""
    gid = meta["id"]
    if tpu_dir and meta.get("tpu_url"):
        for cand in (tpu_dir / f"{gid}.html",):
            if cand.exists():
                html = cand.read_text()
                return (normalize_live(parse_live(html)), "live_headed", html,
                        dt.datetime.now(dt.timezone.utc).isoformat())
    if not refresh and con is not None:
        html, at = db_raw_page(con, gid)
        if html:
            print(f"  [tpu] {gid}: reusing DB raw page (stored {at[:10]})")
            return normalize_live(parse_live(html)), "live_headed", html, at
    if meta.get("tpu_url"):
        try:
            html = fetch_tpu_html(meta["tpu_url"])
            return (normalize_live(parse_live(html)), "live_fetch", html,
                    dt.datetime.now(dt.timezone.utc).isoformat())
        except Exception as e:
            print(f"  [tpu] {gid}: {e!r} -> curated")
    # No TPU source at all: sibling-derived when no reference page exists
    # (tpu_url None by policy), curated fallback otherwise.
    return None, ("sibling" if meta.get("tpu_url") is None else "curated"), None, None


def flag(n: int | None) -> str:
    if not n:
        return "missing"
    return "ok" if n >= MIN_30D_SAMPLES else "thin"


def build_expanded(meta: dict, shs: dict, live, method: str, rel_names: set[str],
                   tpu_at: str | None = None) -> dict:
    gid = meta["id"]
    cur = tpu_curated.CURATED[gid]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    # Variant entries sharing one TPU reference page (e.g. RX 480 4GB on the
    # 8GB page): entry-defining fields win over the live page. Documented in
    # config.py per entry; keeps variant memory/MSRP honest.
    if live is not None and meta.get("tpu_overrides"):
        live = {**live, **meta["tpu_overrides"]}

    def LV(k, default=None):
        v = live.get(k) if live else None
        return default if v is None else v

    # Live-vs-curated verification: every live value is diffed against the
    # baseline; mismatches are stored (and printed) for human review.
    discrepancies: list[str] = []
    if live is not None:
        _CHECK = [
            ("die", "die"), ("architecture", "architecture"), ("process_nm", "process_nm"),
            ("transistors_b", "transistors_b"), ("die_size_mm2", "die_size_mm2"),
            ("sm_count", "sm"), ("cuda_cores", "cuda"), ("tensor_cores", "tensor_cores"),
            ("rt_cores", "rt_cores"), ("tmu", "tmu"), ("rop", "rop"),
            ("base_clock_mhz", "base_mhz"), ("boost_clock_mhz", "boost_mhz"),
            ("memory_clock_effective_gbps", "mem_mhz_effective_gbps"),
            ("memory_size_gb", "memory_gb"), ("memory_type", "memory_type"),
            ("memory_bus_bits", "bus_bits"), ("memory_bandwidth_gbs", "bandwidth_gbs"),
            ("tdp_w", "tdp_w"), ("suggested_psu_w", "psu_w"),
            ("power_connectors", "power_connector"), ("bus_interface", "bus_interface"),
            ("launch_price_usd", "msrp_usd"),
            ("theoretical_fp32_tflops", "fp32_tflops"),
            ("theoretical_fp16_tflops", "fp16_theoretical_tflops"),
            ("theoretical_fp64_gflops", "fp64_gflops"),
            ("matrix_fp16_tflops", "matrix_fp16_dense"),
            ("matrix_bf16_tflops", "matrix_bf16_dense"),
            ("matrix_tf32_tflops", "matrix_tf32_dense"),
            ("matrix_fp8_tflops", "matrix_fp8_dense"),
            ("matrix_fp4_tflops", "matrix_fp4_dense"),
            ("matrix_int8_tops", "matrix_int8_dense"),
            ("matrix_int4_tops", "matrix_int4_dense"),
        ]
        for _lk, _ck in _CHECK:
            _lv, _cv = live.get(_lk), cur.get(_ck)
            if _lv is None:
                continue
            if isinstance(_lv, float) and isinstance(_cv, (int, float)):
                _ok = abs(_lv - _cv) <= max(0.02 * abs(_cv), 0.35)
            else:
                _ok = str(_lv) == str(_cv)
            if not _ok:
                discrepancies.append(f"{_lk}: live={_lv!r} curated={_cv!r}")

    def _I(v):
        return None if v is None else int(v)

    specs = {
        "die": LV("die", cur["die"]), "architecture": LV("architecture", cur["architecture"]),
        "process_nm": _I(LV("process_nm", cur["process_nm"])),
        "transistors_m": _I(round(LV("transistors_b", cur["transistors_b"]) * 1000)) if LV("transistors_b", cur.get("transistors_b")) is not None else None,
        "die_size_mm2": _I(LV("die_size_mm2", cur["die_size_mm2"])),
        "sm": _I(LV("sm_count", cur.get("sm"))), "cuda": _I(LV("cuda_cores", cur.get("cuda"))),
        "tensor": _I(LV("tensor_cores", cur.get("tensor_cores"))),
        "rt": _I(LV("rt_cores", cur.get("rt_cores"))),
        "tmu": _I(LV("tmu", cur.get("tmu"))), "rop": _I(LV("rop", cur.get("rop"))),
        "base_mhz": _I(LV("base_clock_mhz", cur.get("base_mhz"))),
        "boost_mhz": _I(LV("boost_clock_mhz", cur.get("boost_mhz"))),
        "mem_gbps": (lambda v: float(v) if v is not None else None)(LV("memory_clock_effective_gbps", cur.get("mem_mhz_effective_gbps"))),
        "vram_gb": (lambda v: None if v is None else int(round(v)))(LV("memory_size_gb", cur.get("memory_gb"))),
        "mem_type": LV("memory_type", cur.get("memory_type")),
        "mem_bus": _I(LV("memory_bus_bits", cur.get("bus_bits"))),
        "bw": (lambda v: float(v) if v is not None else None)(LV("memory_bandwidth_gbs", cur.get("bandwidth_gbs"))),
        "tdp_w": _I(LV("tdp_w", cur.get("tdp_w"))), "psu_w": _I(LV("suggested_psu_w", cur.get("psu_w"))),
        "power": LV("power_connectors", cur.get("power_connector")),
        "bus_if": LV("bus_interface", cur.get("bus_interface")),
        "msrp_usd": _I(LV("launch_price_usd", cur.get("msrp_usd"))),
        "chip": (live.get("chip") if live else None) or cur["die"].split("-")[0],
        "foundry": LV("foundry"), "process_detail": LV("process"),
        "l2": LV("l2_cache"), "outputs": LV("outputs"), "slot": LV("slot_width"),
        "launch_iso": cur["launch"],
        "tpu_release_iso": LV("release_iso") or parse_iso_date(LV("release_date")),
        "tpu_announced_iso": LV("announced_iso"),
        "generation": LV("generation"), "predecessor": LV("predecessor"),
        "successor": LV("successor"), "production": LV("production"),
        "driver": LV("driver_support"), "l1": LV("l1_cache"),
        "density": float(LV("density_m_per_mm2") or 0) or None,
        "cuda_ver": LV("cuda_version"), "directx": LV("directx"), "opengl": LV("opengl"),
        "opencl": LV("opencl"), "vulkan": LV("vulkan"), "shader": LV("shader_model"),
        "num_vector": LV("numeric_vector"), "num_matrix": LV("numeric_matrix"),
        "len_mm": LV("length_mm"), "hgt_mm": LV("height_mm"), "wid_mm": LV("width_mm"),
        "board_no": LV("board_number"),
        "form_factor": "PCIe AIB", "interconnect": None, "mig": None,
        "currency": "USD", "market": "eBay US via SHS",
    }
    if specs["density"] is None and specs["transistors_m"] and specs["die_size_mm2"]:
        specs["density"] = round(specs["transistors_m"] / specs["die_size_mm2"], 1)

    def _F(live_key, cur_key):
        v = LV(live_key, cur.get(cur_key))
        return float(v) if v is not None else None
    theo = {"f32": _F("theoretical_fp32_tflops", "fp32_tflops"),
            "f16": _F("theoretical_fp16_tflops", "fp16_theoretical_tflops"),
            "f64": _F("theoretical_fp64_gflops", "fp64_gflops"),
            "px": float((live.get("pixel_rate_gpixels") if live else None) or 0) or None,
            "tx": float((live.get("texture_rate_gtexels") if live else None) or 0) or None}
    mx = {"fp4": _F("matrix_fp4_tflops", "matrix_fp4_dense"),
          "fp8": _F("matrix_fp8_tflops", "matrix_fp8_dense"),
          "i4": _F("matrix_int4_tops", "matrix_int4_dense"),
          "i8": _F("matrix_int8_tops", "matrix_int8_dense"),
          "f16": _F("matrix_fp16_tflops", "matrix_fp16_dense"),
          "bf": _F("matrix_bf16_tflops", "matrix_bf16_dense"),
          "tf": _F("matrix_tf32_tflops", "matrix_tf32_dense"),
          # Structured sparsity (2:4) exists on NVIDIA Ampere and newer
          # (Ampere/Ada/Hopper/Blackwell) plus RDNA4; older pages list no
          # sparse note. Curated fallback assumes sparsity only for known
          # sparse-capable architectures, never by default.
          "sparse_mult": 2.0 if (live and live.get("sparse_note")) or
          (not live and cur.get("architecture") in ("Ampere", "Ada Lovelace", "Hopper", "Blackwell", "Blackwell 2.0")) else 1.0}
    def cond_block(d: dict | None):
        if not d:
            return None
        canon = d["api_avg_30d"] if d["api_avg_30d"] is not None else d["windows"]["30d"]["avg"]
        return {"avg": d["windows"]["30d"]["avg"], "api_avg": d["api_avg_30d"], "canon": canon,
                "windows": d["windows"], "monthly": d["monthly"],
                "stats": d["stats"], "product_meta": d["summary"].get("product", {})}

    # Precision selection (METHODOLOGY.md): dense max over available tensor
    # precisions (FP4 > INT4 > INT8/FP8 > FP16/BF16 > TF32 in practice; the
    # winner is recorded). Volta (V100) has FP16-only tensor hardware, so a
    # fixed FP4/INT4 lookup would miss it. Tensor-less cards (no Matrix
    # block) fall back to the highest Theoretical number.
    _cands = {"FP4 dense": mx["fp4"], "INT4 dense": mx["i4"],
              "INT8 dense": mx["i8"], "FP8 dense": mx["fp8"],
              "FP16 dense": mx["f16"], "BF16 dense": mx["bf"],
              "TF32 dense": mx["tf"]}
    _avail = {k: v for k, v in _cands.items() if v is not None}
    if _avail:
        ai_prec, ai_t = max(_avail.items(), key=lambda kv: kv[1])
    else:
        _theo_cands = {"Theoretical FP32 (fallback)": theo["f32"],
                       "Theoretical FP16 (fallback)": theo["f16"],
                       "Theoretical FP64 (fallback)": (theo["f64"] / 1000.0 if theo["f64"] is not None else None)}
        _theo_avail = {k: v for k, v in _theo_cands.items() if v is not None}
        if _theo_avail:
            ai_prec, ai_t = max(_theo_avail.items(), key=lambda kv: kv[1])
        else:
            ai_prec, ai_t = "missing", None
        # No structured sparsity without tensor hardware: fallback is dense-only.
        mx["sparse_mult"] = 1.0
    used_block = cond_block(shs.get("used"))
    new_block = cond_block(shs.get("new"))
    price_u = (used_block or {}).get("canon")
    sm = mx["sparse_mult"] or 1.0
    ai = {"t": ai_t, "ts": round(ai_t * sm, 2) if ai_t is not None else None, "prec": ai_prec,
          "pd": round(ai_t / price_u, 4) if (ai_t and price_u) else None,
          "pds": round(ai_t * sm / price_u, 4) if (ai_t and price_u) else None}

    n30u = (used_block["windows"]["30d"]["n"] if used_block else 0)
    n30n = (new_block["windows"]["30d"]["n"] if new_block else 0)
    dep = ret = None
    if price_u and specs["msrp_usd"]:
        dep = int(round((specs["msrp_usd"] - price_u) / specs["msrp_usd"] * 10000))
        ret = int(round(price_u / specs["msrp_usd"] * 10000))

    tpu_slug = None
    if meta.get("tpu_url"):
        tpu_slug = meta["tpu_url"].rstrip("/").split("/")[-1].split(".")[0]
    disp = meta["name"]
    for _pfx in ("NVIDIA ", "AMD ", "Intel "):
        if disp.startswith(_pfx):
            disp = disp[len(_pfx):]
            break
    # TPU table naming is inconsistent ("SUPER" caps, "12 GB" spaced, plain
    # "RTX 3080" for the 10GB card); explicit aliases live in config.py.
    # Cards absent from the table (3090 Ti, 3080 12GB, 3060/2060 8/12GB
    # refreshes) stay None — never attribute a sibling's score.
    cand = meta.get("relperf_name") or disp
    rel_name = cand if cand in rel_names else None
    # ok = specs came from TPU (live page or verified TPU-derived sibling);
    # curated fallback has no live TPU source behind it.
    status_tpu = "ok" if method in ("live_headed", "live_fetch", "sibling") else "missing"

    return {
        "id": gid, "short_name": meta["short_name"], "full_name": meta["name"],
        "sources": {"tpu_page": meta.get("tpu_id"), "tpu_slug": tpu_slug,
                    "shs_slug": meta["shs_slug"]},
        "provenance": {"tpu_method": method, "tpu_at": tpu_at or now,
                       "shs_at": (shs.get("used") or {}).get("fetched_at") or now},
        "specs": specs, "theoretical": theo, "matrix": mx, "ai": ai,
        "pricing": {"used": used_block, "new": new_block},
        "depreciation": {"dep_bps": dep, "ret_bps": ret},
        "status": {"tpu": status_tpu, "pu": flag(n30u), "pn": flag(n30n)},
        "relperf_name": rel_name,
        "discrepancies": discrepancies,
        "collected_at": now,
        "_raw_shs": shs,  # listings carrier for the DB builder; stripped before store
    }




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated GPU ids (pricing refresh subset)")
    ap.add_argument("--tpu-live-dir", default=None,
                    help="fresh headed-Chromium dumps; needed for NEW GPUs or with --refresh-specs")
    ap.add_argument("--refresh-specs", default=None,
                    help="'all' or comma-separated ids to force fresh TPU specs (default: reuse DB copy)")
    ap.add_argument("--db", default=None, help="database path (default gpu.db)")
    args = ap.parse_args()

    from dbtools import store
    from dbtools.store import connect

    only = set(args.only.split(",")) if args.only else None
    tpu_dir = Path(args.tpu_live_dir) if args.tpu_live_dir else None
    refresh = set(args.refresh_specs.split(",")) if args.refresh_specs else set()
    refresh_all = "all" in refresh
    con = connect(Path(args.db) if args.db else DB_PATH)

    # relperf: fresh dump > any DB raw page > keep existing table (never wipe)
    rel_base = next((tpu_dir / f"{m['id']}.html" for m in GPUS
                     if tpu_dir and (tpu_dir / f"{m['id']}.html").exists()), None)
    if rel_base:
        rel_rows = parse_relperf(rel_base.read_text())
        base_name = rel_base.stem
    else:
        rel_rows, base_name = [], None
        row = con.execute("SELECT html FROM raw_pages LIMIT 1").fetchone()
        if row:
            from dbtools.store import zdecomp_bytes
            rel_rows = parse_relperf(zdecomp_bytes(row[0]))
            mrow = con.execute(
                "SELECT value FROM meta WHERE key='relperf_base'").fetchone()
            base_name = mrow[0] if mrow else None
    if store.put_relperf(con, rel_rows):
        con.execute("INSERT INTO meta VALUES ('relperf_base',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=?", (base_name,) * 2)
    con.execute("INSERT INTO meta VALUES ('relperf_note',?) "
                "ON CONFLICT(key) DO UPDATE SET value=?",
                ("TPU Relative Performance (gaming); each page's own card is implicit at 100.",) * 2)
    rel_names = {n for n, _ in rel_rows} or {
        r[0] for r in con.execute("SELECT name FROM relperf")}

    records = []
    for meta in GPUS:
        if only and meta["id"] not in only:
            continue
        print(f"Fetching {meta['id']} ...")
        live, method, _html, _tat = load_live(
            tpu_dir, meta, con,
            refresh=(refresh_all or meta["id"] in refresh))
        shs = {}
        for cond in ("used", "new"):
            try:
                shs[cond] = fetch_condition(meta["shs_slug"], cond)
            except Exception as e:
                print(f"  [shs] {cond}: {e!r} -> missing")
                shs[cond] = None
        rec = build_expanded(meta, shs, live, method, rel_names, tpu_at=_tat)
        store.replace_pricing(con, meta["id"], shs)
        if _html:
            store.put_raw_page(con, meta["id"], _html,
                               dt.datetime.now(dt.timezone.utc).isoformat())
        records.append(rec)
        u = rec["pricing"]["used"] or {}
        print(f"  staged. AI={rec['ai']['t']} T, used30d=${u.get('canon')} "
              f"(n={u.get('windows', {}).get('30d', {}).get('n')}, flag={rec['status']['pu']}), "
              f"new30d={(rec['pricing']['new'] or {}).get('canon')}, {rec['ai']['pd']} T/$")
        for _d in rec["discrepancies"]:
            print(f"    DIFF vs curated: {_d}")

    records.sort(key=lambda r: (r["ai"]["pd"] or 0), reverse=True)
    import json as _json
    # Summary always reflects the FULL database state (a partial --only run
    # must not shrink the served leaderboard). Manifest tracks fetched set.
    # NOTE: this connection has no row_factory; get_gpu_full needs Row rows.
    import sqlite3 as _sq
    con.row_factory = _sq.Row
    from dbtools import query as _q
    _all = []
    for _m in GPUS:
        try:
            _r = _q.get_gpu_full(con, _m["id"])
        except Exception:
            _r = None
        if _r and _r["pricing"]["used"]:
            _all.append(_r)
    _seen = {r["id"] for r in records}
    _all = [r for r in _all if r["id"] not in _seen] + records
    _all.sort(key=lambda r: (r["ai"]["pd"] or 0), reverse=True)
    _gens = sorted({_g for r in _all for _g in [r["specs"].get("generation")] if _g})
    _cov = " + ".join(g.replace("GeForce ", "RTX ", 1) if g.startswith("GeForce ") else g
                      for g in _gens)
    _cov = _cov or "Discrete GPUs"
    summary = {
        "title": f"AI Compute (dense) per Dollar — {_cov}",
        "coverage": _cov,
        "methodology": ("AI TFLOPS = dense max over available tensor precisions from TechPowerUp Matrix Performance "
                        "(gaming rel-perf, sparsity + full tensor precisions alongside); Ampere "
                        "has no FP4 so INT4 dense wins, Blackwell FP4 wins, Volta FP16 wins. Where Matrix numbers are absent, the "
                        "highest Theoretical number is used. Price = SecondHandSilicon "
                        "trailing-30-day mean of used eBay sold listings (new tracked too); "
                        f"samples below {MIN_30D_SAMPLES} listings are flagged 'thin'. "
                        "Metric = AI TFLOPS / price."),
        "decisions": {"scope": "external accelerator cards (discrete + workstation + datacenter)",
                      "metric": "dense max(FP4, INT4)",
                      "thin_sample_rule": f"flag when 30d n < {MIN_30D_SAMPLES}",
                      "tpu_strategy": "fully live via headed Chromium (playwright-cli)"},
        "sources": ["https://www.techpowerup.com/gpu-specs", "https://secondhandsilicon.com/"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpus": [{"id": r["id"], "short_name": r["short_name"], "ai_tflops": r["ai"]["t"],
                  "price_usd": (r["pricing"]["used"] or {}).get("canon"),
                  "price_new_usd": (r["pricing"]["new"] or {}).get("canon"),
                  "price_flag": r["status"]["pu"],
                  "price_n30d": r["pricing"]["used"]["windows"]["30d"]["n"]
                  if r["pricing"]["used"] else None,
                  "tflops_per_dollar": r["ai"]["pd"], "memory_gb": r["specs"]["vram_gb"],
                  "bandwidth_gbs": r["specs"]["bw"], "tdp_w": r["specs"]["tdp_w"],
                  "bus_interface": r["specs"]["bus_if"]} for r in _all],
    }
    run_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    res = store.commit_run(con, run_ts, records, summary)
    con.close()
    print(f"\ncommitted {run_ts} to {args.db or DB_PATH} "
          f"({len(records)} gpus, history +{res['history_added']})")
    print("view live: ./serve  ->  http://localhost:8000/website/")
    # Loud failure: GPUs with no used-pricing block render unranked. The DB
    # keeps previous rows (never wiped), but the operator must know to retry.
    _missing = sorted(r["id"] for r in records if not (r["pricing"] or {}).get("used"))
    if _missing:
        print("\nWARNING: used price fetch failed for: " + ", ".join(_missing))
        print("  previous pricing (if any) was kept; retry with:")
        print(f"  .venv/bin/python -m scraper.run --only {','.join(_missing)}"
              + (f" --db {args.db}" if args.db else ""))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
