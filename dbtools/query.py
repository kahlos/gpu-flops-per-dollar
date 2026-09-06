"""Reads: gpu.db -> expanded v2-shaped dicts (same shape run.py builds, so the
site bundler and analysis code work unchanged). Transparent zstd handling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .store import zdecomp

D = lambda c: None if c is None else c / 100
D10 = lambda c: None if c is None else c / 10


def open_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def get_gpu_full(con: sqlite3.Connection, gid: str) -> dict | None:
    g = con.execute("SELECT * FROM gpus WHERE id=?", (gid,)).fetchone()
    if not g:
        return None
    g = dict(g)
    specs = {
        "die": g["die"], "architecture": g["architecture"], "process_nm": g["process_nm"],
        "transistors_m": g["transistors_m"], "die_size_mm2": g["die_size_mm2"],
        "sm": g["sm"], "cuda": g["cuda"], "tensor": g["tensor"], "rt": g["rt"],
        "tmu": g["tmu"], "rop": g["rop"], "base_mhz": g["base_mhz"], "boost_mhz": g["boost_mhz"],
        "mem_gbps": D10(g["mem_gbps_x10"]), "vram_gb": g["vram_gb"], "mem_type": g["mem_type"],
        "mem_bus": g["mem_bus"], "bw": D10(g["bw_x10"]),
        "tdp_w": g["tdp_w"], "psu_w": g["psu_w"], "power": g["power"], "bus_if": g["bus_if"],
        "msrp_usd": g["msrp_usd"], "chip": g["chip"], "foundry": g["foundry"],
        "process_detail": g["process_detail"], "l2": g["l2"], "outputs": g["outputs"],
        "slot": g["slot"], "launch_iso": g["launch_iso"],
        "tpu_release_iso": g["tpu_release_iso"], "tpu_announced_iso": g["tpu_announced_iso"],
        "generation": g["generation"], "predecessor": g["predecessor"],
        "successor": g["successor"], "production": g["production"], "driver": g["driver"],
        "l1": g["l1"], "density": D10(g["density_x10"]),
        "cuda_ver": g["cuda_ver"], "directx": g["directx"], "opengl": g["opengl"],
        "opencl": g["opencl"], "vulkan": g["vulkan"], "shader": g["shader"],
        "num_vector": g["num_vector"], "num_matrix": g["num_matrix"],
        "len_mm": g["len_mm"], "hgt_mm": g["hgt_mm"], "wid_mm": g["wid_mm"],
        "board_no": g["board_no"], "form_factor": g["form_factor"],
        "interconnect": g["interconnect"], "mig": g["mig"],
        "currency": g["currency"], "market": g["market"],
    }
    pricing = {}
    for cond in ("used", "new"):
        st = con.execute("SELECT * FROM cond_stats WHERE gpu_id=? AND cond=?",
                         (gid, cond)).fetchone()
        if not st:
            pricing[cond] = None
            continue
        st = dict(st)
        buckets = con.execute(
            "SELECT * FROM monthly WHERE gpu_id=? AND cond=? ORDER BY mkey", (gid, cond))
        monthly = [{"monthKey": r["mkey"], "avgPrice": D(r["avg_c"]), "minPrice": D(r["min_c"]),
                    "maxPrice": D(r["max_c"]), "stdDev": D(r["std_c"]),
                    "totalListings": r["tot"], "newListings": r["new_n"],
                    "usedListings": r["used_n"],
                    "monthChange": r["chg_bps"] / 100 if r["chg_bps"] is not None else None}
                   for r in buckets]
        wins = {r["w"]: {"n": r["n"], "avg": D(r["avg_c"]), "med": D(r["med_c"]),
                         "lo": D(r["lo_c"]), "hi": D(r["hi_c"])}
                for r in con.execute("SELECT * FROM windows WHERE gpu_id=? AND cond=?",
                                     (gid, cond))}
        stats = {"currentPrice": D(st["cur_c"]), "average": D(st["api_avg_c"]),
                 "low": D(st["lo_c"]), "high": D(st["hi_c"]), "count": st["cnt"],
                 "priceChange30d": st["chg_bps"] / 100 if st["chg_bps"] is not None else None,
                 "oneYearHigh": D(st["yr_hi_c"]), "oneYearLow": D(st["yr_lo_c"]),
                 "totalListings": st["tot"]}
        canon = D(st["canon_c"])
        pricing[cond] = {
            "avg": wins.get("30d", {}).get("avg"), "api_avg": D(st["api_avg_c"]),
            "canon": canon, "windows": wins, "monthly": monthly,
            "stats": stats,
            "product_meta": {"benchmarkScore": st["bench"], "brand": st["brand"],
                             "series": st["series"], "basePrice": st["base_price"]},
        }
    u = pricing["used"] or {}
    return {
        "id": gid, "short_name": g["short_name"], "full_name": g["full_name"],
        "sources": {"tpu_page": g["tpu_page"], "tpu_slug": g["tpu_slug"],
                    "shs_slug": g["shs_slug"]},
        "provenance": {"tpu_method": g["tpu_method"], "tpu_at": g["tpu_at"],
                       "shs_at": g["shs_at"]},
        "collected_at": g["collected_at"],
        "specs": specs,
        "theoretical": {"f32": D(g["t_f32"]), "f16": D(g["t_f16"]), "f64": g["t_f64"],
                        "px": D10(g["t_px"]), "tx": D10(g["t_tx"])},
        "matrix": {"fp4": D(g["m_fp4"]), "fp8": D(g["m_fp8"]), "i4": D(g["m_i4"]),
                   "i8": D(g["m_i8"]), "f16": D(g["m_f16"]), "bf": D(g["m_bf"]),
                   "tf": D(g["m_tf"]),
                   "sparse_mult": 1 + g["m_sparse_bps"] / 10000
                   if g["m_sparse_bps"] is not None else None},
        "ai": {"t": D(g["ai_t"]), "ts": D(g["ai_ts"]), "prec": g["ai_prec"],
               "pd": g["ai_pd"] / 10000 if g["ai_pd"] is not None else None,
               "pds": g["ai_pds"] / 10000 if g["ai_pds"] is not None else None},
        "pricing": pricing,
        "depreciation": {"dep_bps": g["dep_bps"], "ret_bps": g["ret_bps"]},
        "status": {"tpu": g["status_tpu"], "pu": g["status_pu"], "pn": g["status_pn"]},
        "relperf_name": g["relperf_name"],
        "discrepancies": json.loads(g["discrepancies"] or "[]"),
    }


def list_ids(con) -> list[str]:
    return [r[0] for r in con.execute("SELECT id FROM gpus ORDER BY id")]


def leaderboard(con) -> list[dict]:
    q = """SELECT g.id, g.short_name, g.ai_t, c.canon_c, g.ai_pd, g.vram_gb, g.bw_x10,
                  g.tdp_w, g.bus_if, g.status_pu
           FROM gpus g LEFT JOIN cond_stats c ON c.gpu_id=g.id AND c.cond='used'
           ORDER BY g.ai_pd DESC"""
    out = []
    for r in con.execute(q):
        out.append({"id": r[0], "short_name": r[1], "ai_tflops": r[2] / 100 if r[2] else None,
                    "price_usd": r[3] / 100 if r[3] is not None else None,
                    "tflops_per_dollar": r[4] / 10000 if r[4] is not None else None,
                    "memory_gb": r[5], "bandwidth_gbs": r[6] / 10 if r[6] else None,
                    "tdp_w": r[7], "bus_interface": r[8], "price_flag": r[9]})
    return out


def get_listings(con, gid: str, cond: str) -> dict:
    """Decoded raw listings: {d0, rows: [[iso_date, price_usd, title], ...]}."""
    import datetime as _dt
    row = con.execute("SELECT d0, data FROM listing_blobs WHERE gpu_id=? AND cond=?",
                      (gid, cond)).fetchone()
    if not row:
        return {"d0": None, "rows": []}
    d0, blob = row
    d = zdecomp(blob)
    y, m, dd = int(d0[:4]), int(d0[5:7]), int(d0[8:10])
    o0 = _dt.date(y, m, dd).toordinal()
    out = []
    for off, cents, ti in d["rows"]:
        dt_s = _dt.date.fromordinal(o0 + off).isoformat()
        out.append([dt_s, cents / 100, d["t"][ti]])
    return {"d0": d0, "rows": out}


def get_relperf(con) -> dict:
    rows = con.execute("SELECT name, pct FROM relperf").fetchall()
    base = con.execute("SELECT value FROM meta WHERE key='relperf_base'").fetchone()
    note = con.execute("SELECT value FROM meta WHERE key='relperf_note'").fetchone()
    return {"base_page": base[0] if base else None, "note": note[0] if note else None,
            "table": [[n, p] for n, p in rows]}


def get_summary(con) -> dict:
    row = con.execute("SELECT summary FROM runs ORDER BY ts DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else {}


def get_history(con, gid: str) -> list[dict]:
    """Full per-run record snapshots, oldest first (deltas resolved).

    Deltas chain onto their immediate predecessor, so a single forward pass
    suffices; any base mismatch falls back to full chain resolution.
    """
    from . import delta as _delta
    from .store import resolve_blob, zdecomp
    out, prev_ts, prev_doc = [], None, None
    for (ts,) in con.execute("SELECT ts FROM history WHERE gpu_id=? ORDER BY ts", (gid,)):
        data, kind, base = con.execute(
            "SELECT data, kind, base_ts FROM history WHERE ts=? AND gpu_id=?",
            (ts, gid)).fetchone()
        if kind is None or kind == "full" or base != prev_ts or prev_doc is None:
            prev_doc = resolve_blob(con, gid, ts)[1]
        else:
            prev_doc = _delta.apply(prev_doc, zdecomp(data))
        prev_ts = ts
        out.append({"ts": ts, "record": prev_doc})
    return out


def get_runs(con) -> list[dict]:
    return [{"ts": r[0], "manifest": json.loads(r[1])}
            for r in con.execute("SELECT ts, manifest FROM runs ORDER BY ts")]
