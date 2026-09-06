"""Compaction for the website bundle: expanded record -> short-key dict + pool.

(Bundle-only pooling; the database itself is normalized instead.)
"""

from __future__ import annotations

from .schema import ENUMS, WINDOWS

C = lambda dollars: None if dollars is None else int(round(dollars * 100))
T = lambda tflops: None if tflops is None else int(round(tflops * 100))
X10 = lambda x: None if x is None else int(round(x * 10))
BPS = lambda pct: None if pct is None else int(round(pct * 100))
I = lambda x: None if x is None else int(x)


class Pool:
    def __init__(self): self.items: list[str] = []; self.idx: dict[str, int] = {}

    def put(self, s: str | None):
        if s is None: return None
        if s not in self.idx: self.idx[s] = len(self.items); self.items.append(s)
        return self.idx[s]


def wrow(w: dict | None):
    if not w or not w.get("n"): return [0, None, None, None, None]
    return [w["n"], C(w["avg"]), C(w["med"]), C(w["lo"]), C(w["hi"])]


def mrow(b: dict):
    return [b["monthKey"], C(b["avgPrice"]), C(b["minPrice"]), C(b["maxPrice"]),
            C(b.get("stdDev") or 0), I(b["totalListings"]), I(b.get("newListings") or 0),
            I(b.get("usedListings") or 0), BPS(b.get("monthChange") or 0)]


def compact_cond(c: dict | None, pool: Pool) -> dict | None:
    if not c: return None
    s = c["stats"]
    return {
        "avg": C(c["avg"]), "api": C(c["api_avg"]),
        "w": {w: wrow(c["windows"].get(w)) for w in WINDOWS},
        "m": [mrow(b) for b in c["monthly"]],
        "chg": BPS(s.get("priceChange30d") or 0),
        "yh": C(s.get("oneYearHigh")), "yl": C(s.get("oneYearLow")),
        "tot": I(s.get("totalListings") or s.get("count")),
        "bn": I((c.get("product_meta") or {}).get("benchmarkScore")),
    }


def compact_record(r: dict, pool: Pool) -> dict:
    P = pool.put
    s = r["specs"]
    spec = {
        "die": P(s["die"]), "arc": P(s["architecture"]), "prm": I(s["process_nm"]),
        "trs": I(s["transistors_m"]), "dsz": I(s["die_size_mm2"]),
        "sm": I(s["sm"]), "cud": I(s["cuda"]), "ten": I(s["tensor"]), "rtc": I(s["rt"]),
        "tmu": I(s["tmu"]), "rop": I(s["rop"]), "bcl": I(s["base_mhz"]), "kcl": I(s["boost_mhz"]),
        "mcl": X10(s["mem_gbps"]), "vgb": I(s["vram_gb"]), "vty": P(s["mem_type"]),
        "bus": I(s["mem_bus"]), "bw": X10(s["bw"]), "tdp": I(s["tdp_w"]), "psu": I(s["psu_w"]),
        "pwr": P(s["power"]), "bif": P(s["bus_if"]), "msrp": I(s["msrp_usd"]),
        "chp": P(s["chip"]), "fdy": P(s["foundry"]), "pdet": P(s["process_detail"]),
        "l2c": P(s["l2"]), "out": P(s["outputs"]), "slt": P(s["slot"]),
        "lch": s["launch_iso"], "rel": s["tpu_release_iso"], "ann": s["tpu_announced_iso"],
        "gen": P(s["generation"]), "pre": P(s["predecessor"]), "suc": P(s["successor"]),
        "prd": P(s["production"]), "drv": P(s["driver"]), "l1c": P(s["l1"]),
        "den": X10(s["density"]), "cuv": P(s["cuda_ver"]), "dxx": P(s["directx"]),
        "ogl": P(s["opengl"]), "ocl": P(s["opencl"]), "vlk": P(s["vulkan"]),
        "shm": P(s["shader"]), "nfv": P(s["num_vector"]), "nfm": P(s["num_matrix"]),
        "len": I(s["len_mm"]), "hgt": I(s["hgt_mm"]), "wid": I(s["wid_mm"]),
        "bdn": P(s["board_no"]), "ff": P(s["form_factor"]), "ic": P(s["interconnect"]),
        "mig": s["mig"], "cur": P(s["currency"]), "mkt": P(s["market"]),
    }
    th, mx, ai = r["theoretical"], r["matrix"], r["ai"]
    return {
        "i": r["id"], "n": P(r["short_name"]), "f": P(r["full_name"]),
        "src": {"t": r["sources"].get("tpu_page"), "ts": P(r["sources"].get("tpu_slug")),
                "s": r["sources"]["shs_slug"]},
        "pv": {"tm": ENUMS["tpu_method"].index(r["provenance"]["tpu_method"]),
               "tat": r["provenance"]["tpu_at"], "sat": r["provenance"]["shs_at"]},
        "sp": spec,
        "th": {"f32": T(th["f32"]), "f16": T(th["f16"]), "f64": I(th["f64"]),
               "px": X10(th["px"]), "tx": X10(th["tx"])},
        "mx": {"fp4": T(mx["fp4"]), "fp8": T(mx["fp8"]), "i4": T(mx["i4"]),
               "i8": T(mx["i8"]), "mf": T(mx["f16"]), "bf": T(mx["bf"]),
               "tf": T(mx["tf"]), "spb": BPS((mx["sparse_mult"] - 1) * 100)
               if mx["sparse_mult"] else None},
        "ai": {"t": T(ai["t"]), "ts": T(ai["ts"]), "pu": P(ai["prec"]),
               "pd": I(round(ai["pd"] * 10000)) if ai["pd"] is not None else None,
               "pds": I(round(ai["pds"] * 10000)) if ai["pds"] is not None else None},
        "pr": {"u": compact_cond(r["pricing"].get("used"), pool),
               "n": compact_cond(r["pricing"].get("new"), pool)},
        "dp": {"dep": r["depreciation"]["dep_bps"], "ret": r["depreciation"]["ret_bps"]},
        "st": {"ts": ENUMS["tpu_status"].index(r["status"]["tpu"]),
               "pu": ENUMS["price_flag"].index(r["status"]["pu"]),
               "pn": ENUMS["price_flag"].index(r["status"]["pn"])},
        "rp": P(r["relperf_name"]),
        "ms": r["discrepancies"],
    }


def pack_listings(slug: str, cond: str, d: dict) -> dict:
    """Columnar rows [day_offset, cents, title_idx] + title dictionary."""
    import datetime as _dt
    dates, prices, titles = d["dates"], d["prices"], d["titles"]
    order = sorted(range(len(dates)), key=lambda k: dates[k])
    uniq: dict[str, int] = {}
    tarr: list[str] = []
    rows = []
    for k in order:
        t = titles[k]
        if t not in uniq: uniq[t] = len(tarr); tarr.append(t)
        rows.append([dates[k], C(prices[k]), uniq[t]])
    if not rows:
        return {"slug": slug, "cond": cond, "d0": None, "rows": [], "t": []}
    d0 = rows[0][0]
    o0 = _dt.date(int(d0[:4]), int(d0[5:7]), int(d0[8:10])).toordinal()
    packed = []
    for dt_s, cents, ti in rows:
        o = _dt.date(int(dt_s[:4]), int(dt_s[5:7]), int(dt_s[8:10])).toordinal() - o0
        packed.append([o, cents, ti])
    return {"slug": slug, "cond": cond, "d0": d0, "rows": packed, "t": tarr}
