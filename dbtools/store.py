"""SQLite store: expanded records -> gpu.db. Bulk payloads as zstd blobs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import zstandard as zstd

from .pack import C, X10, BPS, I, pack_listings

SCHEMA_SQL = Path(__file__).with_name("schema.sql").read_text()
ZSTD_LEVEL = 22
_ZC = zstd.ZstdCompressor(level=ZSTD_LEVEL)
_ZD = zstd.ZstdDecompressor()


def zcomp(obj) -> bytes:
    return _ZC.compress(json.dumps(obj, separators=(",", ":")).encode())


def zdecomp(blob: bytes):
    return json.loads(_ZD.decompress(blob).decode())


def zdecomp_bytes(blob: bytes) -> str:
    return _ZD.decompress(blob).decode()


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)
    con.execute("PRAGMA journal_mode=WAL")
    # forward-migration for DBs created before history deltas existed
    cols = {r[1] for r in con.execute("PRAGMA table_info(history)")}
    if "kind" not in cols:
        con.execute("ALTER TABLE history ADD COLUMN kind TEXT")
        con.execute("ALTER TABLE history ADD COLUMN base_ts TEXT")
        con.execute("UPDATE history SET kind='full'")
        con.commit()
    return con


def _s100(x):
    return None if x is None else int(round(x * 100))


_GPU_ALL = ["short_name", "full_name", "tpu_page", "tpu_slug", "shs_slug",
    "tpu_method", "tpu_at", "shs_at", "collected_at",
    "status_tpu", "status_pu", "status_pn", "relperf_name", "discrepancies",
    "die", "architecture", "process_nm", "transistors_m", "die_size_mm2",
    "sm", "cuda", "tensor", "rt", "tmu", "rop", "base_mhz", "boost_mhz",
    "mem_gbps_x10", "vram_gb", "mem_type", "mem_bus", "bw_x10",
    "tdp_w", "psu_w", "power", "bus_if", "msrp_usd",
    "chip", "foundry", "process_detail", "l2", "outputs", "slot",
    "launch_iso", "tpu_release_iso", "tpu_announced_iso",
    "generation", "predecessor", "successor", "production", "driver", "l1",
    "density_x10", "cuda_ver", "directx", "opengl", "opencl", "vulkan", "shader",
    "num_vector", "num_matrix", "len_mm", "hgt_mm", "wid_mm", "board_no",
    "form_factor", "interconnect", "mig", "currency", "market",
    "t_f32", "t_f16", "t_f64", "t_px", "t_tx",
    "m_fp4", "m_fp8", "m_i4", "m_i8", "m_f16", "m_bf", "m_tf", "m_sparse_bps",
    "ai_t", "ai_ts", "ai_prec", "ai_pd", "ai_pds", "dep_bps", "ret_bps"]


def upsert_gpu(con: sqlite3.Connection, r: dict) -> None:
    s, th, mx, ai = r["specs"], r["theoretical"], r["matrix"], r["ai"]
    cols = ["id"] + _GPU_ALL
    con.execute(
        f"INSERT INTO gpus ({','.join(cols)}) VALUES ({','.join(':' + c for c in cols)}) "
        f"ON CONFLICT(id) DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in _GPU_ALL),
        {
            "id": r["id"], "short_name": r["short_name"], "full_name": r["full_name"],
            "tpu_page": r["sources"].get("tpu_page"), "tpu_slug": r["sources"].get("tpu_slug"),
            "shs_slug": r["sources"]["shs_slug"],
            "tpu_method": r["provenance"]["tpu_method"], "tpu_at": r["provenance"]["tpu_at"],
            "shs_at": r["provenance"]["shs_at"], "collected_at": r["collected_at"],
            "status_tpu": r["status"]["tpu"], "status_pu": r["status"]["pu"],
            "status_pn": r["status"]["pn"],
            "relperf_name": r["relperf_name"],
            "discrepancies": json.dumps(r["discrepancies"]),
            "die": s["die"], "architecture": s["architecture"], "process_nm": I(s["process_nm"]),
            "transistors_m": I(s["transistors_m"]), "die_size_mm2": I(s["die_size_mm2"]),
            "sm": I(s["sm"]), "cuda": I(s["cuda"]), "tensor": I(s["tensor"]), "rt": I(s["rt"]),
            "tmu": I(s["tmu"]), "rop": I(s["rop"]),
            "base_mhz": I(s["base_mhz"]), "boost_mhz": I(s["boost_mhz"]),
            "mem_gbps_x10": X10(s["mem_gbps"]), "vram_gb": I(s["vram_gb"]),
            "mem_type": s["mem_type"], "mem_bus": I(s["mem_bus"]), "bw_x10": X10(s["bw"]),
            "tdp_w": I(s["tdp_w"]), "psu_w": I(s["psu_w"]), "power": s["power"],
            "bus_if": s["bus_if"], "msrp_usd": I(s["msrp_usd"]),
            "chip": s["chip"], "foundry": s["foundry"], "process_detail": s["process_detail"],
            "l2": s["l2"], "outputs": s["outputs"], "slot": s["slot"],
            "launch_iso": s["launch_iso"], "tpu_release_iso": s["tpu_release_iso"],
            "tpu_announced_iso": s["tpu_announced_iso"],
            "generation": s["generation"], "predecessor": s["predecessor"],
            "successor": s["successor"], "production": s["production"], "driver": s["driver"],
            "l1": s["l1"], "density_x10": X10(s["density"]),
            "cuda_ver": s["cuda_ver"], "directx": s["directx"], "opengl": s["opengl"],
            "opencl": s["opencl"], "vulkan": s["vulkan"], "shader": s["shader"],
            "num_vector": s["num_vector"], "num_matrix": s["num_matrix"],
            "len_mm": I(s["len_mm"]), "hgt_mm": I(s["hgt_mm"]), "wid_mm": I(s["wid_mm"]),
            "board_no": s["board_no"], "form_factor": s["form_factor"],
            "interconnect": s["interconnect"], "mig": s["mig"],
            "currency": s["currency"], "market": s["market"],
            "t_f32": _s100(th["f32"]), "t_f16": _s100(th["f16"]), "t_f64": I(th["f64"]),
            "t_px": X10(th["px"]), "t_tx": X10(th["tx"]),
            "m_fp4": _s100(mx["fp4"]), "m_fp8": _s100(mx["fp8"]), "m_i4": _s100(mx["i4"]),
            "m_i8": _s100(mx["i8"]), "m_f16": _s100(mx["f16"]), "m_bf": _s100(mx["bf"]),
            "m_tf": _s100(mx["tf"]),
            "m_sparse_bps": BPS((mx["sparse_mult"] - 1) * 100) if mx["sparse_mult"] else None,
            "ai_t": _s100(ai["t"]), "ai_ts": _s100(ai["ts"]), "ai_prec": ai["prec"],
            "ai_pd": I(round(ai["pd"] * 10000)) if ai["pd"] is not None else None,
            "ai_pds": I(round(ai["pds"] * 10000)) if ai["pds"] is not None else None,
            "dep_bps": r["depreciation"]["dep_bps"], "ret_bps": r["depreciation"]["ret_bps"],
        })


def replace_pricing(con: sqlite3.Connection, gpu_id: str, shs: dict) -> None:
    for cond, key in (("used", "used"), ("new", "new")):
        d = shs.get(key)
        con.execute("DELETE FROM monthly WHERE gpu_id=? AND cond=?", (gpu_id, cond))
        con.execute("DELETE FROM windows WHERE gpu_id=? AND cond=?", (gpu_id, cond))
        con.execute("DELETE FROM cond_stats WHERE gpu_id=? AND cond=?", (gpu_id, cond))
        con.execute("DELETE FROM listing_blobs WHERE gpu_id=? AND cond=?", (gpu_id, cond))
        if not d:
            continue
        for b in d["monthly"]:
            con.execute(
                "INSERT INTO monthly VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (gpu_id, cond, b["monthKey"], C(b["avgPrice"]), C(b["minPrice"]),
                 C(b["maxPrice"]), C(b.get("stdDev") or 0), I(b["totalListings"]),
                 I(b.get("newListings") or 0), I(b.get("usedListings") or 0),
                 BPS(b.get("monthChange") or 0)))
        for w, s in d["windows"].items():
            con.execute(
                "INSERT INTO windows VALUES (?,?,?,?,?,?,?,?)",
                (gpu_id, cond, w, s["n"], C(s["avg"]), C(s["med"]), C(s["lo"]), C(s["hi"])))
        st, pm = d["stats"], d["summary"].get("product", {})
        api = st.get("average") or st.get("currentPrice")
        canon = api if api is not None else d["windows"]["30d"]["avg"]
        con.execute(
            """INSERT INTO cond_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (gpu_id, cond, C(api), C(st.get("currentPrice")), C(st.get("low")), C(st.get("high")),
             I(st.get("count")), BPS(st.get("priceChange30d") or 0),
             C(st.get("oneYearHigh")), C(st.get("oneYearLow")),
             I(st.get("totalListings")), I(pm.get("benchmarkScore")),
             pm.get("brand"), pm.get("series"), I(pm.get("basePrice")), C(canon),
             d["fetched_at"]))
        packed = pack_listings(pm.get("id") or gpu_id, cond,
                               {"dates": d["dates"], "prices": d["prices"], "titles": d["titles"]})
        import hashlib
        sha = hashlib.sha256(json.dumps(packed, sort_keys=True).encode()).hexdigest()[:16]
        con.execute("INSERT INTO listing_blobs VALUES (?,?,?,?,?,?,?)",
                    (gpu_id, cond, packed["d0"], len(packed["rows"]), len(packed["t"]),
                     zcomp(packed), sha))


def put_raw_page(con, gpu_id: str, html: str, fetched_at: str) -> bool:
    """Store a fresh raw page; skips the write when the bytes are unchanged."""
    import hashlib
    raw = html.encode()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    cur = con.execute("SELECT sha FROM raw_pages WHERE gpu_id=?", (gpu_id,)).fetchone()
    if cur and cur[0] == sha:
        return False
    con.execute("INSERT INTO raw_pages VALUES (?,?,?,?) ON CONFLICT(gpu_id) DO UPDATE SET "
                "html=excluded.html, sha=excluded.sha, fetched_at=excluded.fetched_at",
                (gpu_id, _ZC.compress(raw), sha, fetched_at))
    return True


def put_relperf(con, rows: list) -> bool:
    """Replace the shared table only when we actually have rows (never wipe)."""
    if not rows:
        return False
    con.execute("DELETE FROM relperf")
    con.executemany("INSERT INTO relperf VALUES (?,?)", [(n, p) for n, p in rows if n])
    return True


def record_sha(expanded: dict) -> str:
    """Content hash for change detection. Run timestamps (collected_at,
    provenance.tpu_at/shs_at) are EXCLUDED — runs are timestamped in `runs`;
    history must only grow when the DATA changes."""
    import hashlib, copy
    doc = copy.deepcopy(expanded)
    doc.pop("_raw_shs", None)
    doc.pop("collected_at", None)
    prov = doc.get("provenance")
    if isinstance(prov, dict):
        prov.pop("tpu_at", None)
        prov.pop("shs_at", None)
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


_MAX_CHAIN = 20  # store a fresh full snapshot after this many deltas


def resolve_blob(con, gid: str, ts: str | None = None):
    """Resolve a history blob to its full document, walking delta chains."""
    from . import delta as _delta
    if ts is None:
        row = con.execute(
            "SELECT ts FROM history WHERE gpu_id=? ORDER BY ts DESC LIMIT 1", (gid,)).fetchone()
        if not row:
            return None, None
        ts = row[0]
    ops_chain = []
    cur = ts
    while True:
        row = con.execute(
            "SELECT data, kind, base_ts FROM history WHERE ts=? AND gpu_id=?",
            (cur, gid)).fetchone()
        if not row:
            raise KeyError(f"history blob missing: {gid} @ {cur}")
        data, kind, base = row
        payload = zdecomp(data)
        if kind is None or kind == "full":
            doc = payload
            break
        ops_chain.append(payload)
        cur = base
        if len(ops_chain) > 500:
            raise ValueError(f"delta chain too long for {gid}")
    for ops in reversed(ops_chain):
        doc = _delta.apply(doc, ops)
    return ts, doc


def _store_history(con, ts: str, gid: str, slim: dict, h: str) -> str:
    """Store full snapshot or delta vs latest state. Returns kind stored."""
    from . import delta as _delta
    prev_ts, prev_doc = resolve_blob(con, gid)
    if prev_doc is None:
        kind, base, payload = "full", None, slim
    else:
        depth, cur = 0, prev_ts
        while True:
            row = con.execute(
                "SELECT kind, base_ts FROM history WHERE ts=? AND gpu_id=?",
                (cur, gid)).fetchone()
            if not row or row[0] in (None, "full"):
                break
            cur, depth = row[1], depth + 1
        if depth >= _MAX_CHAIN:
            kind, base, payload = "full", None, slim
        else:
            kind, base, payload = "delta", prev_ts, _delta.diff(prev_doc, slim)
    con.execute("INSERT INTO history VALUES (?,?,?,?,?,?) ON CONFLICT(ts,gpu_id) DO NOTHING",
                (ts, gid, zcomp(payload), h, kind, base))
    return kind


def commit_run(con, ts: str, expanded_records: list[dict], summary: dict) -> dict:
    """Upsert everything; history only for changed records (dedup across runs)."""
    manifest, added = {}, 0
    for r in expanded_records:
        upsert_gpu(con, r)
        h = record_sha(r)
        manifest[r["id"]] = h
        # Gate on the GPU's OWN latest blob, not the latest run manifest:
        # partial (--only) runs must not rewrite GPUs whose data is unchanged
        # since their last stored snapshot (timestamps are excluded from sha).
        cur = con.execute(
            "SELECT sha FROM history WHERE gpu_id=? ORDER BY ts DESC LIMIT 1",
            (r["id"],)).fetchone()
        if cur and cur[0] == h:
            continue
        slim = dict(r)
        slim.pop("_raw_shs", None)
        _store_history(con, ts, r["id"], slim, h)
        added += 1
    con.execute("INSERT INTO runs VALUES (?,?,?)", (ts, json.dumps(summary), json.dumps(manifest)))
    con.execute("INSERT INTO meta VALUES ('schema','1') ON CONFLICT(key) DO UPDATE SET value='1'")
    con.execute("INSERT INTO meta VALUES ('last_run',?) ON CONFLICT(key) DO UPDATE SET value=?",
                (ts, ts))
    con.commit()
    # keep the deliverable a single file: checkpoint WAL back into the db
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.execute("PRAGMA journal_mode=DELETE")
    con.commit()
    return {"history_added": added, "manifest": manifest}
