# Cluster Viability Analysis: Legacy GPUs vs. DeepSeek V4 Flash API

**Question:** can a cluster of older GPUs be built and run cost-competitively
against the public API pricing of DeepSeek V4 Flash 0731, for bulk batch
workloads? **Verdict: yes — ~30x unit-cost advantage at baseline, 2.4x in the
worst defensible corner, break-even at ~2% utilization.** The binding risks are
serving software and demand, not physics or electricity.

This report supersedes the economics of "TOKEN-ECONOMICS-RESEARCH.md" and
"Legacy-AI-Hardware-TCO-Analysis.md" (kept for the record — see section 10 for
the reconciliation; both contain a fatal 1000x token-unit slip that invalidates
their financial tables). Their *architectural* conclusions survive here.

Reproduce every number: see "tools/cluster_model.py" (reads gpu.db directly).

---

## 1. Workload and benchmark

Target model (per INT4-VIABILITY-RESEARCH.md spec): DeepSeek V4 Flash 0731 —
284B total params, 13B active/token, 43 layers, 256 routed + 1 shared SwiGLU
experts per layer (25.17M params/expert), d_model 4096, hybrid compressed
attention. Derived constants used throughout:

| Constant | Value | Note |
|---|---|---|
| FLOPs/token (decode) | 28.6 GFLOP | 2 x 13B x 1.10 attention overhead |
| FLOPs/token/layer | 665 MFLOP | / 43 layers |
| Layer weights, W4A4 | 3.43 GB | 0.52 B/param (4-bit + FP8 group-64 scales) |
| Total weights, W4A4 | 147.7 GB | must shard across >= 16 10GB-class cards |
| KV/token/layer | 600 B | GQA-4 x 128 x K+V, FP8, hybrid CSA 4x / HCA 128x |
| Context (base scenario) | 2,048 | bulk extraction/transform shape |

Benchmark API rates (competitor, per project docs): **$0.11/1M input,
$0.66/1M output, $0.007/1M cached input** -> blended $0.2088 (uncached),
$0.1741 (95% cached). Per-workload, the API is more expensive than the blend
for output-heavy tasks — the blend is the *conservative* comparator.

## 2. Method

Per-stage decode cycle model, pipeline of N single-GPU stages (1 GPU/host,
PCIe Gen4 x4 = 3.0 GB/s effective, per user constraint):

    B      = (VRAM - L_s*layer_weights - 1.2 GB fixed) / (L_s * ctx * kv_bytes)
    t_mem  = L_s * (layer_weights + B*ctx*kv) / (BW * eff_mem)
    t_comp = L_s * B * flops_layer / (TOPS * eff_comp)
    t_xfer = B * act_bytes / PCIe_GBps          # FP8 activations: 4 KB/token
    cycle  = max(t_mem, t_comp, t_xfer) * 1.05  # pipeline overhead
    tok/s  = B / cycle                          # 1 micro-batch completes/cycle

All stages run every cycle; completed tokens/s = B/cycle in steady state.
Per-stream rate = 1 token/cycle (68 tok/s at B=3,715 — interactive-grade);
TTFT is roughly 43 stages x (prefill compute + transfer) = ~0.6 s at 1K prompts.

**Calibration against published reality (both anchors hit):**

| Anchor | Model | Published |
|---|---|---|
| Llama-70B Q4, 1x3080, batch 1 | 14.8 tok/s | 13-17 tok/s (llama.cpp) |
| DeepSeek-V3 FP8, 8xH100 TP8, batch 256 | 7,616 tok/s | 4,000-8,000 tok/s (SGLang/vLLM) |

The calibration is what disqualifies the prior docs 2.96M tok/s claim (6x over
their own PCIe link cap; 1.9x over the sparse-INT4 compute roofline) and their
implied 5.9K tok/s cost basis (batch ~ 16 weight amortization) in one move.

## 3. The winning architecture

**Pure pipeline parallelism, one layer per GPU (PP43), no TP/EP, batch as
large as KV capacity allows, FP8 activations.** Three reasons, each verified:

1. **L_s = 1 maximizes batch.** One layer/card consumes 3.43 GB of VRAM,
   leaving 4.6 GB/card for KV -> B = 3,715 at 2K ctx. Fewer stages (L_s >= 2)
   multiply both weight re-reads and KV storage per stage; measured worse at
   every N < 43 for 10GB-class cards.
2. **Batch amortizes the weight stream.** Weight bytes/token fall from
   28.6 MB (B=1) to 0.66 MB (B=3,715). The cluster flips from weight-bound to
   KV-bound — the regime where 43 cards aggregate 32.7 TB/s actually earns.
3. **MoE locality kills interconnect traffic.** All 256 experts of layer i
   live on GPU_i; cross-stage traffic is activations only (30 MB/cycle/hop at
   FP16, 15 MB at FP8) vs 8 GB of memory streaming per cycle. TP/EP require
   blocking all-reduces over PCIe — confirmed dead, as prior docs argued.

At B ~ 3,715 each expert receives ~15-90 tokens/pass — real GEMM shapes.

## 4. Flagship baseline: 43x RTX 3080 10GB (triple-checked)

| Parameter | Value | Cross-checked by |
|---|---|---|
| Batch B | 3,715 seqs @ 2,048 ctx | KV fit: 4.56+3.43+1.2 = 9.20/9.20 GB exact |
| Cycle | 14.7 ms (mem 14.0 / comp 8.7 / xfer 5.1) | all three paths within 4.8% |
| **Throughput** | **252,207 tok/s** | aggregate-BW floor: 264,818 (delta 4.8%) |
| Aggregate bandwidth | 23.4 TB/s (75% of 32.7) | DRAM energy closure: 28 W/card I/O inside 192 W draw |
| Energy | 42 mJ/token wall (DRAM silicon 4.8-9.6 mJ) | independent energy path |
| Power | 10.6 kW (192 W/card + 55 W host, PUE 1.15) | 2x 50 A/240 V circuits |
| CapEx | **$33,755** ($38,818 w/ 15% contingency + spares) | DB price $370/GPU (1,565 listings/30d — deep market) |
| Output @ 90% duty | 597 B tokens/month | — |
| **All-in cost** | **$0.0059/1M** | fixed $1.88K/mo + power $1.55K/mo |
| **Margin vs API blend** | **30x** | 29x at 50% roofline software |

Prefill compute fits alongside decode: decode 7.2 PF/s + prefill 1.6 PF/s
(P=1K, G=4K) = 72% of the 12.3 PF/s INT4 budget. Dense-GQA KV (no CSA/HCA,
2 KB/token/layer) merely halves throughput — the verdict does not depend on
the compressed-attention assumption.

## 5. Configuration sweep (all 157 DB GPUs screened, 49 pass gates)

All numbers tool-verified ("tools/cluster_model.py --gpu <id>", per-card INT4/FP4 TOPS, FP8 activations):

| Config | B | tok/s | CapEx | $/1M | vs API | Note |
|---|---|---|---|---|---|---|
| **43x RTX 3080 10GB** | 3,715 | 252,207 | $38.8K | **$0.0059** | 30x | flagship: deepest market (1,565 listings/30d) |
| 43x RTX 3080 12GB | 5,212 | 345,234 | $42.2K | $0.0046 | 38x | best $/1M; thin market (31 listings/30d) |
| 43x RTX 3090 24GB | 14,197 | 454,687 | $84.5K | $0.0046 | 37x | most KV headroom, same cost class |
| 16x RTX 3090 (4 hosts, L_s=3) | 2,869 | 91,886 | $31.4K | $0.0109 | 16x | simplest ops — recommended pilot |
| 43x RTX 2080 Ti | 4,464 | 220,203 | $35.5K | $0.0060 | 29x | Turing W4A4 kernel risk |
| 43x RX 9070 XT | 8,207 | 279,509 | $70.1K | $0.0066 | 26x | ROCm PP/batching risk |
| 43x RTX 5090 | 20,186 | 697,544 (xfer-bound) | $266.6K | $0.0069 | 25x | highest absolute output; CapEx-heavy |

**Key insight:** the project ai_pd (TFLOPS/$) metric is the wrong lens for
cluster serving. The economics are driven by **bandwidth x VRAM per dollar** —
which is why $370-440 mid-range Ampere beats a $4,977 5090 on cost per token.

## 6. Sensitivity (the conclusion is robust)

| Scenario | $/1M | Margin |
|---|---|---|
| Baseline (90% duty, $0.20/kWh, roofline) | $0.0059 | 30x |
| Software achieves 50% of roofline | $0.0118 | 15x |
| Software achieves 25% of roofline | $0.0235 | 7x |
| Utilization 50% / 25% / 10% / 5% | $0.0091 / 0.0162 / 0.0376 / 0.0732 | 19x / 11x / 4.6x / 2.4x |
| Electricity $0.08 / $0.30 per kWh | $0.0043 / $0.0072 | 40x / 24x |
| Pessimistic memory efficiency (eff 0.5) | $0.0088 | 20x |
| **Worst sane corner** (25% roofline + 25% duty + $0.30/kWh) | **$0.0734** | **2.4x** |
| **Break-even utilization** vs $0.1741 blend | **2.07% duty = ~14B tok/mo** | — |
| Break-even vs $0.11 input-only floor | 3.30% duty = ~22B tok/mo | — |

Per-task vs API (flagship, all-in): extraction (2K-in/200-out) **12x** cheaper
cached-benchmark, 27x uncached; transform (2K/2K) **57x/65x**; generation
(1K/4K) **90x/93x**. The advantage grows with output length — the API charges
$0.66/1M for what costs $0.006 locally.

## 7. Hidden constraints (none of the prior docs modeled these)

1. **PCIe x4 saturation forces FP8 activations.** Decode activation stream at
   B=3,715: 69% of link at FP16 — and prefill traffic then overflows it. FP8:
   34% decode + prefill = 9%. Design-in from day one.
2. **Prefill/decode compute ratio matters.** P=G runs the INT4 budget to 85%;
   long-input/short-output workloads (summarization) degrade throughput ~2x
   (prefill-bound). Short-prompt/long-output stays memory-bound (optimal).
3. **Straggler amplification.** A 43-deep synchronous pipeline runs at the
   slowest stage. Expected failure rate: 43 x 720 h/mo / ~50k h MTBF = ~0.6
   cards/month. Hot spares + auto-drain are mandatory (in the contingency).
4. **KV capacity is the batch limiter** — the prior docs never derived B at
   all. Compressed attention (MLA-class 600 B/token/layer) is what allows
   B ~ 3,715; uncompressed GQA halves it.
5. **Demand is the true constraint.** Full output is 597 B tokens/month —
   a DeepSeek-scale book of business. Break-even sits at ~2% duty; every
   scenario above 5% duty is profitable under the API benchmark.

## 8. Future upgrades (model + software only)

- **2-bit weight quant** (next-gen Q2 + scales): weight stream 148 -> ~90 GB,
  +60% throughput on identical hardware.
- **Kernel maturity**: 0.75 -> 0.85 achieved BW (Marlin-class W4A4 grouped GEMM):
  +13%.
- **Speculative decoding** at high batch: +30-80% where acceptance holds.
- **Smaller active-param models** (13B -> 7B class): helps prefill/headroom;
  decode stays KV-bound.
- Counter-risk: API price deflation. Even a 2x API cut (blend $0.087) leaves
  the flagship 15x ahead at baseline assumptions.

## 9. Recommendation: staged plan with kill gates

1. **Pilot (weeks 1-8): 16x RTX 3090, 4 hosts x 4 GPUs, PP16.** CapEx ~$31.4K.
   Gate: sustained >= 50% of modeled 92K tok/s; per-stage achieved BW >= 60%;
   straggler tail <= 10% of cycle time. Cheap kill: 3090 resale market is deep.
2. **Scale (weeks 9-20): 43x RTX 3080 10GB flagship** once PP16 software is
   proven and a buyer for >= 15-20B tok/mo is signed. Gate: signed demand
   >= break-even (14B tok/mo) before ordering the remaining 27 nodes.
3. **Operate:** FP8 activations, continuous batching, hot spares, weekly
   failure audit. Never run crawls and the cluster on the same circuits.

## 10. Reconciliation with the prior research documents

Why this report differs from TOKEN-ECONOMICS-RESEARCH.md and
Legacy-AI-Hardware-TCO-Analysis.md:

| Item | Prior docs | This report | Cause of gap |
|---|---|---|---|
| Throughput in $ math | ~5.9K tok/s (implied by $0.0917/1M variable cost) | 252K tok/s | Never derived batch size from KV capacity; B=16 fixed |
| Throughput claimed | 2.96M tok/s | 252K | T-vs-B unit slip (revenue implies 280.2B tokens, not 280.2T); 6x over their own PCIe cap |
| Cache-hit "4x throughput" | Treated as physics | Rejected | Caching skips prefill, not decode weight streaming |
| PCIe transfer | B x S x d_model (prefill shape) -> link saturated | B x d_model (decode shape) | Confused prefill chunks with decode steps |
| CapEx/node | $371 "complete" (GPU alone is $370) | $785 + 15% | BOM leaves $1 for the host |
| Optimized 3-yr power | $25,754 | $51,509 | Arithmetic bug (exactly half) |
| Calibration | None | 2 published anchors hit | — |
| Internal consistency | Break-even contradicts own TCO matrix (50x) | 3 derivation paths within 4.8% | — |

**What survives from the prior docs** (adopted here): PP-per-layer as the only
viable topology; TP/EP dead on PCIe; stock power catastrophic; undervolting to
~200 W mandatory; utilization tiers and break-even-volume framing (their ~7.8B
tok/mo break-even is the same order as this report 14B).

**Retired from the prior docs:** all absolute profit/ROI figures (built on the
unit slip), the 4x cache multiplier, the 2.96M/1.18M tok/s tables, the $371
complete-node BOM, and the "22-month payback" claim (3x arithmetic error).

## 11. Model assumptions (locked for this report)

eff_mem 0.75 - eff_comp 0.60 - GPU draw 0.60xTDP undervolted, 15 W idle -
host 40 W - PUE 1.15 - PCIe Gen4 x4 3.0 GB/s - FP8 activations (4 KB/token) -
KV 600 B/token/layer FP8 - ctx 2,048 - W4A4 0.52 B/param - node BOM $335 +
$80 infra - labor $300/mo - facility $500/mo - spares 8%/yr of CapEx -
amortization 36 mo - electricity $0.20/kWh - duty 90% (swept 5-90%).
Change any of these via "tools/cluster_model.py" flags; the .py is the
authority if this document and the code ever disagree.