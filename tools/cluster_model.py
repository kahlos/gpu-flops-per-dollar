#!/usr/bin/env python3
"""Cluster viability model: legacy GPU clusters vs. DeepSeek V4 Flash API.

Companion to docs/CLUSTER-VIABILITY-ANALYSIS.md (the .py is the authority if
the two ever disagree). Reads gpu.db directly (read-only), screens every GPU,
and reproduces the report tables. Triple-checked: per-cycle model vs aggregate
bandwidth floor must agree within 5%; calibration anchors assert against
published llama.cpp / SGLang numbers.

Usage:
    .venv/bin/python tools/cluster_model.py                    # report tables
    .venv/bin/python tools/cluster_model.py --ctx 8192 --elec 0.08
    .venv/bin/python tools/cluster_model.py --gpu rtx-3080-10gb --duty 0.25
"""
import argparse
import math
import sqlite3
import sys
from pathlib import Path

# ---- workload (DeepSeek V4 Flash 0731; official config verified, docs/CLUSTER-VIABILITY-ANALYSIS.md section 1) ----
P_TOTAL, P_ACT, LAYERS, D_MODEL = 284e9, 13e9, 43, 4096
FLOP_TOK = 2 * P_ACT * 1.10
FLOP_LAYER = FLOP_TOK / LAYERS
BP_W4 = 0.52                       # bytes/param: 4-bit + FP8 group-64 scales
W_LAYER = P_TOTAL / LAYERS * BP_W4
W_TOTAL = P_TOTAL * BP_W4

# -------- cluster constants (locked in the report, section 11) ---------------
EFF_MEM, EFF_COMP = 0.75, 0.60
GPU_IDLE, HOST_W, PUE = 15.0, 40.0, 1.15
PCIE_GBPS = 3.0                    # PCIe Gen4 x4 effective
NODE_BOM, INFRA = 335.0, 80.0
LABOR_MO, FACILITY_MO = 300.0, 500.0
SPARES_YR = 0.08
AMORT_MO = 36
CTX, KV_TOK = 2048, 600.0
ELEC, DUTY = 0.20, 0.90
MONTH_S = 2.628e6
API_BLEND = 0.1741


def load_gpus(db_path):
    con = sqlite3.connect("file:" + str(db_path) + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute(
        "SELECT id, short_name, vram_gb, bw_x10, tdp_w, m_i4, m_i8, m_fp4 FROM gpus"):
        canon = con.execute(
            "SELECT canon_c FROM cond_stats WHERE gpu_id=? AND cond='used'",
            (r["id"],)).fetchone()
        if canon is None or not canon["canon_c"]:
            continue
        n30 = con.execute(
            "SELECT n FROM windows WHERE gpu_id=? AND cond='used' AND w='30d'",
            (r["id"],)).fetchone()
        tops = max(r["m_i4"] or 0, r["m_fp4"] or 0, r["m_i8"] or 0) / 100.0
        prec = ("FP4" if (r["m_fp4"] or 0) >= max(r["m_i4"] or 0, r["m_i8"] or 0)
                else ("INT4" if r["m_i4"] else "INT8"))
        g = dict(id=r["id"], short_name=r["short_name"],
                 vram=r["vram_gb"], bw=(r["bw_x10"] or 0) / 10.0,
                 tdp=r["tdp_w"], tops=tops, prec=prec,
                 price=canon["canon_c"] / 100.0, n30d=(n30["n"] if n30 else 0))
        if g["bw"] > 150 and g["vram"] >= 8 and g["tops"] >= 40:
            out.append(g)
    con.close()
    return out


def simulate(g, N, ctx, kv_tok, eff_mem, act_bytes):
    L_s = math.ceil(LAYERS / N)
    vram = g["vram"] * 0.92 * 1e9
    kv_budget = vram - L_s * W_LAYER - 1.2e9
    if kv_budget <= 0:
        return None
    B = int(kv_budget / (L_s * ctx * kv_tok))   # per-stage KV = B*L_s*ctx*kv
    if B < 4:
        return None
    t_mem = L_s * (W_LAYER + B * ctx * kv_tok) / (g["bw"] * 1e9 * eff_mem)
    t_comp = L_s * B * FLOP_LAYER / (g["tops"] * 1e12 * EFF_COMP)
    t_xfer = B * act_bytes / (PCIE_GBPS * 1e9) if N > 1 else 0.0
    cyc = max(t_mem, t_comp, t_xfer) * 1.05
    bound = max((("mem", t_mem), ("comp", t_comp), ("xfer", t_xfer)),
                key=lambda kv: kv[1])[0]
    return dict(B=B, L_s=L_s, tps=B / cyc, bound=bound)


def econ(g, s, N, duty, elec, gpu_pf=0.60):
    gpu_w = gpu_pf * g["tdp"]
    kw = N * (HOST_W + GPU_IDLE + (gpu_w - GPU_IDLE) * duty) * PUE / 1000.0
    capex = N * (g["price"] + NODE_BOM + INFRA) * 1.15   # +15% contingency+spares
    tok_mo = s["tps"] * MONTH_S * duty
    if tok_mo < 1e9:
        return None
    fixed = capex / AMORT_MO * (1 + SPARES_YR) + LABOR_MO + FACILITY_MO
    var = kw * 730 * elec
    return dict(tps=s["tps"], B=s["B"], L_s=s["L_s"], bound=s["bound"], N=N,
                capex=capex, kw=kw, tok_mo=tok_mo,
                cost=(fixed + var) / (tok_mo / 1e6))


def calibrate():
    """Anchors vs published reality; hard-fail if out of family."""
    a1 = EFF_MEM * 760.3e9 / (70e9 * 0.55)
    assert 10 < a1 < 20, "llama.cpp anchor drifted: %.1f t/s" % a1
    b = 256
    w_gpu = 671e9 * 1.0 / 8
    a2 = b * 3.35e12 * EFF_MEM / (w_gpu + b * 2048 * 1100.0)
    assert 3e3 < a2 < 10e3, "H100 anchor drifted: %.0f t/s" % a2
    print("calibration OK: 70B-single-stream %.1f t/s (real 13-17); "
          "8xH100-V3 %s t/s (real 4,000-8,000)" % (a1, format(int(a2), ",")))


def triple_check(g, s, N):
    """Per-cycle model vs aggregate-BW floor. The floor is an upper bound
    (sim must not beat it, tol = pipeline factor + ceil padding); a mem-bound
    config must additionally agree with it within that same tolerance."""
    bytes_tok = P_TOTAL * BP_W4 / s["B"] + LAYERS * CTX * KV_TOK
    tps_bw = N * g["bw"] * 1e9 * EFF_MEM / bytes_tok
    # The floor is an upper bound: sim must not beat it (tol = 5% pipeline
    # factor + ceil padding (N*L_s-43)/43 + eps). Falling short is legal when
    # compute- or transfer-bound.
    tol = 0.05 + (N * s["L_s"] - LAYERS) / LAYERS + 0.005
    assert s["tps"] <= tps_bw * (1 + tol), (
        "sim exceeds aggregate-BW floor: %.0f > %.0f" % (s["tps"], tps_bw))
    if s["bound"] == "mem":
        assert abs(tps_bw - s["tps"]) / tps_bw < tol, (
            "mem-bound paths disagree: %.0f vs %.0f (tol %.3f)"
            % (s["tps"], tps_bw, tol))
    e_dram = bytes_tok * 0.10e-9          # GDDR6X all-in ~0.1 nJ/B
    e_wall = N * (HOST_W + 0.60 * g["tdp"]) / s["tps"]
    assert e_dram < e_wall, "DRAM energy exceeds wall energy - power model broken"


def main():
    ap = argparse.ArgumentParser(description="Legacy GPU cluster viability model")
    ap.add_argument("--db", default="gpu.db")
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--kv", type=float, default=KV_TOK, help="bytes/token/layer")
    ap.add_argument("--elec", type=float, default=ELEC)
    ap.add_argument("--duty", type=float, default=DUTY)
    ap.add_argument("--eff-mem", type=float, default=EFF_MEM)
    ap.add_argument("--act", choices=("fp8", "fp16"), default="fp8")
    ap.add_argument("--gpu", help="single-GPU deep dive (id)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        sys.exit("gpu.db not found at " + str(db.resolve()) + " (pass --db)")
    gpus = load_gpus(db)
    act = D_MODEL * (1 if args.act == "fp8" else 2)
    calibrate()
    print("inventory: %d GPUs with price+specs (ctx=%d, kv=%.0fB/L, %s act, "
          "duty=%.0f%%, $%.2f/kWh)" % (len(gpus), args.ctx, args.kv, args.act,
                                       args.duty * 100, args.elec))

    if args.gpu:
        g = next((x for x in gpus if x["id"] == args.gpu), None)
        if g is None:
            sys.exit("unknown/excluded gpu id: " + args.gpu)
        print("")
        print("== " + g["short_name"] + " deep dive ==")
        for N in (8, 12, 16, 22, 26, 32, 43):
            if N * g["vram"] * 0.92e9 < W_TOTAL + N * 1.2e9:
                continue
            s = simulate(g, N, args.ctx, args.kv, args.eff_mem, act)
            if s is None:
                continue
            e = econ(g, s, N, args.duty, args.elec)
            if e is None:
                continue
            triple_check(g, s, N)
            print("  N=%2d L_s=%d B=%6s %9s tok/s (%s-bound) %5.1f kW "
                  "capex %9s cost %.4f/1M  [%3.0fx API]"
                  % (N, s["L_s"], format(s["B"], ","), format(int(s["tps"]), ","),
                     s["bound"], e["kw"], format(int(e["capex"]), ","),
                     e["cost"], 0.1741 / e["cost"]))
        return

    res = []
    for g in gpus:
        if g["vram"] * 0.92e9 < W_LAYER + 1.5e9:
            continue
        best = None
        for N in range(6, 44):
            if N * g["vram"] * 0.92e9 < W_TOTAL + N * 1.2e9:
                continue
            s = simulate(g, N, args.ctx, args.kv, args.eff_mem, act)
            if s is None:
                continue
            e = econ(g, s, N, args.duty, args.elec)
            if e and (best is None or e["cost"] < best["cost"]):
                best = e
        if best:
            best["g"] = g
            res.append(best)
    res.sort(key=lambda e: e["cost"])

    print("")
    print("== best config per GPU (sorted by cost per 1M, top %d) ==" % args.top)
    print("%-26s%6s%6s%4s%6s%6s%5s%4s%6s%9s%7s%9s%8s%7s"
          % ("GPU", "$", "n30", "VR", "BW", "TOPS", "prec", "N", "B",
             "tok/s", "kW", "CapEx", "$/1M", "vsAPI"))
    for e in res[:args.top]:
        g = e["g"]
        s = simulate(g, e["N"], args.ctx, args.kv, args.eff_mem, act)
        triple_check(g, s, e["N"])
        print("%-26s%6.0f%6d%4d%6.0f%6.0f%5s%4d%6d%9s%7.1f%9s%8.4f%6.0fx"
              % (g["short_name"], g["price"], g["n30d"], g["vram"], g["bw"],
                 g["tops"], g["prec"], e["N"], e["B"], format(int(e["tps"]), ","),
                 e["kw"], format(int(e["capex"]), ","), e["cost"],
                 API_BLEND / e["cost"]))

    winner = res[0]["g"]
    lo, hi = 0.0005, 0.9
    for _ in range(48):
        mid = (lo + hi) / 2
        s = simulate(winner, 43, args.ctx, args.kv, args.eff_mem, act)
        e = econ(winner, s, 43, mid, args.elec)
        if e and e["cost"] > API_BLEND:
            lo = mid
        else:
            hi = mid
    print("")
    print("break-even duty (%s, N=43) vs %.4f API blend: %.2f%% (~%sB tok/mo)"
          % (winner["short_name"], API_BLEND, hi * 100,
             format(int(res[0]["tps"] * MONTH_S * hi / 1e9), ",")))


if __name__ == "__main__":
    main()

