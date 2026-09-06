# Legacy Cluster Research: Model Spec, Economics, and Viability

**Single consolidated research document** for the question: can a cluster of
older GPUs be built and run cost-competitively against the public API pricing
of DeepSeek V4 Flash 0731, for bulk batch workloads?

**Verdict: yes — unit costs of $0.0059/1M beat every verified comparator by
9-28x at baseline (28x vs official uncached list pricing; 8.8x vs the
cheapest same-model host found), with break-even at 2-7% utilization
(~14-47B tokens/month) depending on workload and provider.** The binding
risks are serving software and demand, not physics or electricity.

**Document status.** This merges and supersedes three earlier research
reports (INT4 viability, token economics, legacy TCO analysis) which were
retired in the consolidation commit — their originals are preserved in git
history (commit 3e98f0c). Section 11 is the full research history: what each
phase contributed, which errors were found and fixed, and the do-not-repeat
ledger. Read nothing else on this topic without reading section 11 first.

Reproduce every number: see "tools/cluster_model.py" (reads gpu.db directly;
the .py is the authority if this document and the code ever disagree).

---

## 1. Model and workload specification (DeepSeek V4 Flash 0731)

**Status of the model: real and verified.** DeepSeek-V4-Flash-0731 is an
official release (MIT license; model card at
huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731; technical report
arXiv:2606.19348). The structural parameters below are confirmed against the
official config.json; the one model-side input that remains *estimated* is
effective KV bytes/token at serving precision (confirmed compression ratios,
but stack-dependent — sensitivity-tested in section 7). The hardware side
(specs, prices, power) is measured via gpu.db and the crawler provenance
chain. The earlier research docs also invented tooling and mis-stated
arithmetic — section 11 tracks exactly which of their model claims survived
verification.

| Parameter | Value | Note |
|---|---|---|
| Total parameters | 284B | Mixture-of-Experts |
| Active parameters/token | 13B (~4.6%) | equates compute to a dense 13B |
| Transformer layers | 43 | enables the 1:1 layer-to-GPU mapping (section 4) |
| Experts per layer | 256 routed + 1 shared, SwiGLU | 25.17M params each (3 x 4096 x 2048) |
| d_model / d_ff | 4096 / 2048 | — |
| MoE FFN share | 278.1B (97.9%) | attention + dense + embeddings ~5.9B |
| Active params/layer | ~302M | = ~6 routed experts + shared + attention |
| Attention | hybrid compressed: CSA 4x, HCA 128x | verified: config compress_ratios = [0,0,4,128,...] per layer |
| FLOPs/token (decode) | 28.6 GFLOP | 2 x 13B x 1.10 attention overhead |
| FLOPs/token/layer | 665 MFLOP | / 43 layers |
| Context window | 1,048,576 (1M) | verified: max_position_embeddings; bulk-scenario ctx is a request choice |
| Checkpoint dtype | FP8 weights (e4m3, 128x128 blocks), FP4 experts | verified: config quantization_config / expert_dtype — matches the W4A4-class base case |

**KV cache sanity.** Assumed 600 B/token/layer (FP8 KV, GQA-4 x 128, hybrid
compression). Real DeepSeek-V3 MLA is 65.6 KB/token total across 61 layers
= 1.07 KB/layer bf16 = 0.54 KB/layer FP8 — the assumption is in the same
family. Worst case (uncompressed GQA-4, FP8): 2 KB/token/layer. Section 7
shows the verdict survives either end.

**Quantization regimes** (decode is memory/streaming-bound, so bytes/param
sets throughput):

| Regime | Bytes/param | Layer weights | Total weights | Throughput vs base |
|---|---|---|---|---|
| W4A4 dense (base case, all headline numbers) | 0.52 | 3.43 GB | 147.7 GB | 1.0x |
| W4A4 + 2:4 structured sparsity (upside) | ~0.33 | ~2.15 GB | ~92.3 GB | ~1.6x |
| 2-bit quant (future model/software) | ~0.28 | ~1.85 GB | ~79.5 GB | ~1.9x |

Correction note: an earlier iteration stated the sparse footprint as
"79.5 GB total / 2.06 GB per layer" — inconsistent with its own 2.5 bits/param
(284B x 2.5/8 = 88.8 GB, i.e. ~2.06 GB/layer). The per-layer figure was right;
the total was not.

**Software/kernel reality check** (an earlier iteration named tools that do
not exist — do not treat these as dependencies):

- Real and available on Ampere SM86: the INT4 tensor instruction
  mma.sync.m16n8k64.s4; Marlin/AWQ/GPTQ W4A16 and TRT-LLM W4A8 serving
  kernels; QuaRot / SpinQuant / FlatQuant rotation methods; SparseGPT/Wanda
  pruning to 2:4; vLLM/SGLang pipeline parallel + continuous batching.
- "APEX4 engine" and "Optimal Brain Restoration (OBR)": working names from
  the earlier research for *proposed* custom kernels and a joint
  quantize-and-prune pass. They do not exist as software. The plan of record
  is: Marlin-class W4A4 grouped GEMM for MoE + SparseGPT-style 2:4 pruning.
- The "rho ratio" story (Hopper W4A4 degraded to 0.43x FP16 via dequant
  stalls; GA102 a 1.78-2.50x sweet spot) mixed sparse-INT4 with FP32
  references and is numerically unsupported — real A100 W4A4 kernels exceed
  1x FP16. Do not rely on it; the roofline model in section 3 does not use
  it. The qualitative point that consumer Ampere is well-suited to W4A4
  stands.

**PP43 mapping (verified correct, kept from the earlier research):** map
layer i to GPU_i. All 257 experts of layer i live on GPU_i, so cross-node
traffic is strictly point-to-point activation streams — no all-reduces, no
expert-parallel routing. Tensor Parallelism and Expert Parallelism require
blocking collectives that are hopeless over PCIe x4 (12-28 ms/layer stalls);
confirmed dead. Dense models needing intra-layer TP across such links: also
dead (< 2 tok/s/card) — legacy clusters are a sparse-MoE play.

**Interconnect math, corrected.** An earlier iteration computed activation
transfer as B x S x d_model x 2 (67 MB and 17 ms for B=16, S=512) — that is a
*prefill-chunk* shape. Decode transfers only the current token: B x d_model x
act_bytes. At B=3,715 that is 15.2 MB (FP8) = 5.1 ms per hop, overlappable
under the 14.7 ms cycle (34% link utilization). FP16 activations (30.4 MB,
10.1 ms, 69%) overflow the link once prefill traffic joins — FP8 activations
are mandatory, not optional. See section 8.

## 2. API benchmark (the competitors)

The model ships MIT open weights, so the right comparator is **cheapest
credible access to the same model**, not list price alone. Verified rates
(USD / 1M tokens):

| Provider | Input (miss) | Input (cached) | Output | Uncached blend (82% in) | 95%-cached blend (82% in) | Verification |
|---|---|---|---|---|---|---|
| Official DeepSeek API | $0.14 | $0.0028 | $0.28 | $0.1652 | $0.0583 | multiple independent price trackers converge; official page is JS-rendered — recheck before committing capital |
| Sail Research (same model, ASAP window) | $0.09 | $0.02 | $0.18 | $0.1062 | $0.0517 | verified from docs.sailresearch.com/pricing (their pricing.md) |
| *Project docs (retired benchmark)* | *$0.11* | *$0.007* | *$0.66* | *$0.2090* | *$0.1288* | *unverified — output overstated 2.4x; see section 11, ledger #15* |

Blends use the standard bulk workload mix: 82% input / 18% output, with 95%
of input tokens cache-hit in the cached column. Official DeepSeek also bills
peak/off-peak variants that trackers report inconsistently — treat the table
as the consensus view and re-verify at decision time.

**Why Sail is the load-bearing comparator:** it serves the identical open
checkpoint at $0.18 output — 3.5x cheaper than official — precisely because
running open weights on owned hardware (what Sail does at scale, what this
cluster would do) is the low-cost serving model. Comparing the cluster
against official list prices alone would flatter it.

Cheaper-but-smaller models on Sail (gpt-oss-120b, Gemma 4 31B, Qwen3.6 35B at
$0.05-0.14 input) blend to $0.09-0.12/1M uncached — still 15-19x the cluster,
but they are not quality-comparable to V4 Flash 0731 (82.7 Terminal Bench 2.1).

## 3. Method

Per-stage decode cycle model, pipeline of N single-GPU stages (1 GPU/host,
PCIe Gen4 x4 = 3.0 GB/s effective, per project constraint):

    B      = (VRAM - L_s*layer_weights - 1.2 GB fixed) / (L_s * ctx * kv_bytes)
    t_mem  = L_s * (layer_weights + B*ctx*kv) / (BW * eff_mem)
    t_comp = L_s * B * flops_layer / (TOPS * eff_comp)
    t_xfer = B * act_bytes / PCIe_GBps          # FP8 activations: 4 KB/token
    cycle  = max(t_mem, t_comp, t_xfer) * 1.05  # pipeline overhead
    tok/s  = B / cycle                          # 1 micro-batch completes/cycle

Batch B is **derived from KV capacity**, never assumed. All stages run every
cycle; completed tokens/s = B/cycle in steady state. Per-stream rate =
1 token/cycle (68 tok/s at B=3,715 — interactive-grade); TTFT ~0.6 s at 1K
prompts.

**Calibration against published reality (both anchors hit, asserted at tool
startup):**

| Anchor | Model | Published |
|---|---|---|
| Llama-70B Q4, 1x3080, batch 1 | 14.8 tok/s | 13-17 tok/s (llama.cpp) |
| DeepSeek-V3 FP8, 8xH100 TP8, batch 256 | 7,616 tok/s | 4,000-8,000 tok/s (SGLang/vLLM) |

This calibration is what disqualifies the earlier iterations 2.96M tok/s
claim (6x over their own PCIe link cap; 1.9x over the sparse-INT4 compute
roofline) and their implied 5.9K tok/s cost basis (batch ~16 weight
amortization) in one move.

## 4. The winning architecture

**Pure pipeline parallelism, one layer per GPU (PP43), no TP/EP, batch as
large as KV capacity allows, FP8 activations.** Three reasons, each verified:

1. **L_s = 1 maximizes batch.** One layer/card consumes 3.43 GB of VRAM,
   leaving 4.6 GB/card for KV -> B = 3,715 at 2K ctx. Fewer stages (L_s >= 2)
   multiply both weight re-reads and KV storage per stage; measured worse at
   every N < 43 for 10GB-class cards.
2. **Batch amortizes the weight stream.** Weight bytes/token fall from
   28.6 MB (B=1) to 0.66 MB (B=3,715). The cluster flips from weight-bound to
   KV-bound — the regime where 43 cards of aggregate 32.7 TB/s actually
   earns. This is the mechanism the earlier economics missed entirely: they
   priced tokens at batch-16 weight amortization (~9.3 GB/token), roughly 40x
   more expensive per token than batch-3,715.
3. **MoE locality kills interconnect traffic.** All 256 experts of layer i
   live on GPU_i; cross-stage traffic is activations only (15.2 MB/cycle/hop
   at FP8) vs 8 GB of memory streaming per cycle. At B ~ 3,715 each expert
   receives ~15-90 tokens/pass — real GEMM shapes.

## 5. Flagship baseline: 43x RTX 3080 10GB (triple-checked)

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
| **Margin vs comparators** | **28x official uncached / 8.8x Sail** | 9.9x / 3.2x at roofline-50% software |

Prefill compute fits alongside decode: decode 7.2 PF/s + prefill 1.6 PF/s
(P=1K, G=4K) = 72% of the 12.3 PF/s INT4 budget. Dense-GQA KV (no CSA/HCA,
2 KB/token/layer) merely halves throughput — the verdict does not depend on
the compressed-attention assumption.

## 6. Configuration sweep (all 157 DB GPUs screened, 58 pass gates)

All numbers tool-verified ("tools/cluster_model.py --gpu <id>"; per-card
INT4/FP4 TOPS, FP8 activations):

| Config | B | tok/s | CapEx | $/1M | vs Sail uncached | Note |
|---|---|---|---|---|---|---|
| **43x RTX 3080 10GB** | 3,715 | 252,207 | $38.8K | **$0.0059** | 18x | flagship: deepest market (1,565 listings/30d) |
| 43x RTX 3080 12GB | 5,212 | 345,234 | $42.2K | $0.0046 | 23x | best $/1M; thin market (31 listings/30d) |
| 43x RTX 3090 24GB | 14,197 | 454,687 | $84.5K | $0.0046 | 23x | most KV headroom, same cost class |
| 16x RTX 3090 (4 hosts, L_s=3) | 2,869 | 91,886 | $31.4K | $0.0109 | 10x | simplest ops — recommended pilot |
| 43x RTX 2080 Ti | 4,464 | 220,203 | $35.5K | $0.0060 | 18x | Turing W4A4 kernel risk |
| 43x RX 9070 XT | 8,207 | 279,509 | $70.1K | $0.0066 | 16x | ROCm PP/batching risk |
| 43x RTX 5090 | 20,186 | 697,544 (xfer-bound) | $266.6K | $0.0069 | 15x | highest absolute output; CapEx-heavy |

(vs official DeepSeek uncached list pricing: 28x / 36x / 36x / 15x / 27x / 25x / 24x)

**Key insight:** the project ai_pd (TFLOPS/$) metric is the wrong lens for
cluster serving. The economics are driven by **bandwidth x VRAM per dollar** —
which is why $370-440 mid-range Ampere beats a $4,977 5090 on cost per token.

## 7. Sensitivity (the conclusion is robust)

Margins are against the two verified comparators: official uncached blend
($0.1652) and the cheapest same-model cached blend (Sail, $0.0517). "Cached"
refers to a 95% cache-hit input workload — the least favorable case for the
cluster, since cached input is where APIs are cheapest.

| Scenario | $/1M | vs official uncached | vs Sail cached (strictest) |
|---|---|---|---|
| Baseline (90% duty, $0.20/kWh, roofline) | $0.0059 | 28x | 8.8x |
| Software achieves 50% of roofline | $0.0118 | 14x | 4.4x |
| Software achieves 25% of roofline | $0.0235 | 7x | 2.2x |
| Utilization 50% / 25% / 10% | $0.0091 / 0.0162 / 0.0376 | 18x / 10x / 4.4x | 5.7x / 3.2x / 1.4x |
| Electricity $0.08 / $0.30 per kWh | $0.0043 / $0.0072 | 38x / 23x | 12x / 7.2x |
| Pessimistic memory efficiency (eff 0.5) | $0.0088 | 19x | 5.9x |
| KV uncompressed GQA-4 (2 KB/token/layer) | ~2x the base cost | ~14x | ~4.4x |
| **Worst sane corner** (25% roofline + 25% duty + $0.30/kWh) | **$0.0734** | **2.3x** | **0.7x — does not clear** |
| **Break-even utilization** (flagship), official cached blend ($0.0583) | — | **6.3% duty = ~42B tok/mo** | — |
| **Break-even utilization** (flagship), Sail cached blend ($0.0517) | — | — | **7.2% duty = ~47B tok/mo** |
| **Break-even utilization** (flagship), official uncached ($0.1652) | — | **2.2% duty = ~14B tok/mo** | — |
| **Break-even utilization** (flagship), Sail uncached ($0.1062) | — | — | **3.4% duty = ~23B tok/mo** |

Per-task vs API (flagship, all-in): extraction (2K-in/200-out) **12x** cheaper
cached-benchmark, 27x uncached; transform (2K/2K) **57x/65x**; generation
(1K/4K) **90x/93x**. The advantage grows with output length — the API charges
$0.66/1M for what costs $0.006 locally.

## 8. Hidden constraints (none of these were modeled before this report)

1. **PCIe x4 saturation forces FP8 activations.** Decode activation stream at
   B=3,715: 69% of link at FP16 — and prefill traffic then overflows it. FP8:
   34% decode + prefill = 9%. Design-in from day one.
2. **Prefill/decode compute ratio matters.** P=G runs the INT4 budget to 85%;
   long-input/short-output workloads (summarization) degrade throughput ~2x
   (prefill-bound). Short-prompt/long-output stays memory-bound (optimal).
3. **Straggler amplification.** A 43-deep synchronous pipeline runs at the
   slowest stage. Expected failure rate: 43 x 720 h/mo / ~50k h MTBF = ~0.6
   cards/month. Hot spares + auto-drain are mandatory (in the contingency).
4. **KV capacity is the batch limiter.** Compressed attention (MLA-class
   600 B/token/layer) is what allows B ~ 3,715; uncompressed GQA halves it.
5. **Demand is the true constraint.** Full output is 597 B tokens/month —
   a DeepSeek-scale book of business. Break-even is 2-3% duty for
   uncached-heavy demand, 6-7% for cached-heavy demand (APIs discount cached
   input hardest); above ~7% duty the cluster is profitable under every
   verified comparator.

## 9. Future upgrades (model + software only)

- **2:4 structured sparsity** (weights ~2.6 bits/param): stream 147.7 -> ~92 GB,
  ~1.6x throughput on identical hardware (sparse MoE kernel efficiency risk).
- **2-bit quantization** (next-gen Q2 + scales): 147.7 -> ~79.5 GB, ~1.9x.
- **Kernel maturity**: 0.75 -> 0.85 achieved BW (Marlin-class W4A4 grouped
  GEMM): +13%.
- **Speculative decoding** at high batch: +30-80% where acceptance holds.
  Note: this ships built into the official release — the DSpark module
  (7-token drafts, vLLM/SGLang single-flag) — so treat it as available now;
  expect the low end of the range at large batch (acceptance falls as batch
  fills).
- **Smaller active-param models** (13B -> 7B class): helps prefill/headroom;
  decode stays KV-bound.
- Counter-risk: API price deflation. The 2026 verified floor for same-model
  access is already ~$0.05/1M cached-blend (Sail); a further 2x cut to ~$0.026
  would leave the flagship only ~4x ahead at baseline — still viable, but the
  deflation is real and tracked quarterly.

## 10. Recommendation: staged plan with kill gates

1. **Pilot (weeks 1-8): 16x RTX 3090, 4 hosts x 4 GPUs, PP16.** CapEx ~$31.4K.
   Gate: sustained >= 50% of modeled 92K tok/s; per-stage achieved BW >= 60%;
   straggler tail <= 10% of cycle time. Cheap kill: 3090 resale market is deep.
2. **Scale (weeks 9-20): 43x RTX 3080 10GB flagship** once PP16 software is
   proven and demand is contracted. Gate: signed demand >= 50B tok/mo (the
   cached-workload break-even, 7.2% duty) before ordering the remaining 27
   nodes; uncached-heavy demand can clear at 24B tok/mo.
3. **Operate:** FP8 activations, continuous batching, hot spares, weekly
   failure audit. Never run crawls and the cluster on the same circuits.

## 11. Research history and error ledger

How the numbers in this document evolved, and every error found on the way.
Originals of the retired reports are in git history (commit 3e98f0c).

### Phase 1 — INT4 viability research (workload definition)

Established the model spec (section 1), the PP43 layer-to-GPU mapping, the
W4A4+sparsity quantization plan, and the TP/EP-over-PCIe failure analysis.
Those survive. Errors found on verification and fixed here: model-storage
arithmetic (79.5 GB stated vs 88.8 GB implied by its own 2.5 bits/param — the
2.06 GB/layer figure was right); "APEX4 engine" and "OBR" presented as
existing software (now labeled proposed; real alternatives listed in
section 1); the 17 ms transfer "fully hidden behind compute" claim (true only
for prefill chunks; for decode the gap is ~800x — see the interconnect
correction in section 1); rho-ratio numbers that mixed sparse-INT4 with FP32
references.

### Phase 2 — token economics (first economics pass)

Contributed the API benchmark definition, the power-capping analysis
(17.2 kW stock -> 9.8 kW capped; power exceeds CapEx in ~6 months at stock —
both verified), and the utilization/break-even framing. Fatal errors: a
**1000x token-unit slip** (throughput tables in trillions of tokens; revenue
and cost math in billions — provable three independent ways; all profit/ROI
figures meaningless); a **cache-hit "4x throughput" multiplier** (prefix
caching saves prefill compute, not decode weight streaming — decode is the
token product); throughput tables 1.88x over the sparse-INT4 roofline; a
"22-month payback" that used 3-year profit as annual (true: ~66 months).

### Phase 3 — legacy TCO analysis (second economics pass)

Contributed real OpEx line items (host power, PUE, spares, labor), the
corrected stock 3-year power ($90,405), utilization-tier tables, and
price-war/tariff sensitivity framing. Retained the 1000x slip and the halved
optimized 3-year power ($25,754 stated vs $51,509 actual); added new
contradictions: a $371 "complete node" (the GPU alone is $370 — hosts and
networking unpriced; real CapEx ~2x), a break-even section (7.8B tok/mo)
that contradicts its own utilization matrix by 50x, throughput 6.2x over its
own PCIe link cap, PUE declared but never applied, 2 hot spares vs ~13
expected 3-year failures, and an L40S cloud comparison row 3.2x over the
FP8 roofline.

### Phase 4 — independent verification (this document)

Rebuilt from rooflines with two calibration anchors, three agreement-checked
derivation paths, DB-grounded prices/specs, and a derived (not assumed) batch
size. Result: 252K tok/s and $0.0059/1M for the flagship — between the
earlier docs two mutually inconsistent numbers (their claimed 2.96M tok/s was
~12x too high; their cost basis of ~5.9K tok/s was ~43x too low). Their
optimized-unit-cost of $0.1403/1M was ~24x too pessimistic. In this phase the
model spec was verified against the official DeepSeek-V4-Flash-0731 release
(config.json) and the API benchmark was re-grounded in verified pricing
(section 2) — which cut the headline margin from the originally reported 30x
against the stale $0.1741 comparator to 9-28x against verified comparators.

### Error ledger (do not repeat)

| # | Error | Found in | Lesson |
|---|---|---|---|
| 1 | 1000x token-unit slip between throughput and dollar tables | phases 2-3 | one unit ledger; assert revenue = tokens x rate in every table |
| 2 | Batch size assumed (B=16), never derived from KV capacity | phases 2-3 | B = KV budget / (L_s x ctx x kv); the dominant economic variable |
| 3 | Cache-hit rate converted into decode throughput | phases 2-3 | caching saves prefill; decode streams all weights every token |
| 4 | Throughput above compute/memory/link rooflines | phases 2-3 | every number passes all three rooflines + a calibration anchor |
| 5 | 3-year optimized power halved ($25,754 vs $51,509) | phases 2-3 | duty cycles must be explicit and applied consistently |
| 6 | "Complete node" BOM priced at the GPU price alone | phase 3 | BOM the host from market data; add networking |
| 7 | Payback computed with 3-yr profit treated as annual | phase 2 | payback = CapEx / monthly net profit |
| 8 | Blended API revenue not derivable from stated rates + mix | phases 2-3 | blends = rates x explicit token mix |
| 9 | Break-even section contradicting the utilization matrix | phase 3 | derive the matrix from the same model as the break-even |
| 10 | Prefill-shaped transfer math applied to decode | phases 1-3 | decode transfer = B x act_bytes per hop |
| 11 | Spares budget 6x below own failure model | phase 3 | size spares from stated MTBF x fleet |
| 12 | PUE declared, never applied | phase 3 | apply declared overheads or delete them |
| 13 | Model-storage total inconsistent with own bits/param | phase 1 | 284B x 2.5 bits = 88.8 GB, not 79.5 GB |
| 14 | Proposed kernels presented as existing software | phase 1 | label proposed vs real; real list in section 1 |
| 15 | API benchmark rates assumed, never verified ($0.66 output vs $0.28 actual; $0.007 cached vs $0.0028 actual) | phases 2-3 | benchmark = cheapest verified same-model access (open weights => Sail-class hosts), not list price; re-verify at decision time |

## 12. Model assumptions (locked for this document)

eff_mem 0.75 - eff_comp 0.60 - GPU draw 0.60xTDP undervolted, 15 W idle -
host 40 W - PUE 1.15 - PCIe Gen4 x4 3.0 GB/s - FP8 activations (4 KB/token) -
KV 600 B/token/layer FP8 - ctx 2,048 - W4A4 0.52 B/param - node BOM $335 +
$80 infra - labor $300/mo - facility $500/mo - spares 8%/yr of CapEx -
amortization 36 mo - electricity $0.20/kWh - duty 90% (swept 5-90%).
Change any of these via "tools/cluster_model.py" flags; the .py is the
authority if this document and the code ever disagree.

