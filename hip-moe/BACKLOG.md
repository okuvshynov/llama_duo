# Backlog — deferred experiments and follow-ups

Open items with enough context to pick up cold. Closed work lives in
README.md; this file holds what we chose not to do yet.

## ~~Mixed CPU+GPU EP: the umbrella hypothesis~~ DONE 2026-08-20

Measured (README "the umbrella, measured"): pairs hide while their summed
serial cost stays under the wave — C=1 at n≥2 and C=2 at n≥6 cost
+0.6-3.5%; one pair past capacity costs +33-76%. Both quantitative
predictions were wrong: a lone cold expert is ~600 µs (46-72 GB/s — one
35 MB stream never reaches the 92-98 GB/s wall), and same-expert pairs
are NOT free (~250 µs marginal): repack forward_mul_mat_id is one gemv
per routed row, no grouped gemm (repack.cpp:4498). Viability criterion:
K ≲ 20 of 384 under uniform routing (E[C] and capacity both linear in n);
the trade only clearly wins with real routing skew. Follow-ups spawned
below.

## cpu_dma_latency: does C6 exit eat the umbrella? (from the tcp-latency study)

Cross-pollination from `~/projects/measures/tcp-latency` (2026-08-22, same
Mac Pro pair): holding `/dev/cpu_dma_latency` at 0 collapsed the idle-path
network RTT (idle ping 127 → 35 µs, netperf max 253 → 91) while leaving
the busy-loop mean unchanged — sparse work pays C-state wakeups,
continuous work doesn't. The umbrella has the same shape on-box: during a
GPU wave the cv-parked cold worker and the slept GOMP threads are idle,
and this machine's C6 costs **133 µs exit / 600 µs target residency**
(`/sys/devices/system/cpu/cpu0/cpuidle/state3`) — waves at n≥4
(860-1840 µs) exceed the residency target, so menu can pick C6 and a miss
may open with up to ~133 µs of wake before the first byte of expert
weights moves, against a ~550-600 µs pair budget. Existing data it could
partly explain: the solo placement lottery (488-1003 µs, 46-72 GB/s) and
pair cost sitting above the streaming estimate.

Plan — instrument, not tg32 (the ~10% load-to-load floor can't resolve
this; moe-ep-bench's interleaved A/B resolves ~0.5%):

1. **Confirm the sleep happens at all**: delta
   `/sys/devices/system/cpu/cpu*/cpuidle/state3/usage` (or turbostat)
   across a 50-rep `--cpu-experts` run. If C6 entries ≈ 0 during the
   bench loop, expect a null here and the effect belongs only to sparse
   serving (inter-token idle at 15 t/s), not to the bench.
2. **A/B**: `measures/tcp-latency/hold-cpu-dma-latency.py` held (root)
   vs not. Can't interleave rep-by-rep across a root process — alternate
   whole invocations, ≥3 per arm. Cells: the C=1/C=2 rows (pick K from
   `--probe`, the K=16-caught-nothing lesson) plus the solo price list.
3. **Read tails, not means**: the tcp result was mean-neutral and
   tail-collapsing; per-rep cpu-wall spread and the solo min-max band are
   where 133 µs would show.
4. **Confounder from the same study**: the tcp work left this box's CPU
   governor on `performance` (non-persistent, survives until reboot) —
   record the governor with every arm, since prior umbrella numbers were
   taken under a different setting.
5. **If it wins**: interacts with "Serving-mode wait policy" below —
   blocking sync frees a core precisely by letting it sleep, so measure
   that config with and without the cap before adopting either. Cost of
   the cap: cores floor at C1, idle power rises; the always-on policy
   decision lives with tcp-latency's persistence item, not here.

## Cold-expert follow-ups (from the umbrella measurement)

- Grouped-gemm mul_mat_id on the CPU repack path: when multiple rows hit
  one expert, the kernel could stream weights once (gemm) instead of once
  per row (gemv). Would cut the C>1 same-expert cost ~in half and is an
  upstream-shaped change — check whether ggml's plain (non-repack) CPU
  mul_mat_id already does this before proposing.
- Real-routing coldness: the design goal is miss TOLERANCE (cold experts
  rarely hit; hide 1-2 misses when they happen), and the umbrella
  measurement prices exactly that — a within-capacity miss is +0.6-3.5%
  on its step, amortized cost = hit rate x ~30-50 µs. What's missing is
  the hit rate: on the real DS-V4-Flash (or Pro) collect per-expert hit
  histograms over a corpus; the coldest-K tail's cumulative hit rate then
  gives mean tax and P(over capacity) directly, and licenses K larger
  than the uniform-routing ceiling (~20). The chat.sh server + a hooked
  ids dump would do it. Deferred indefinitely (2026-08-20): the histogram
  is specific to each model and even each quant of a model, so it belongs
  to a concrete deployment, not to this proof of concept — the PoC
  numbers above (per-miss price, capacity, over-capacity bound) are the
  transferable result.
- Scheduler config is part of the result: default GOMP (no pinning, no
  passive) was the only configuration that worked under overlap;
  GOMP_CPU_AFFINITY=0-15 starves the main thread (+133% instrument tax),
  OMP_WAIT_POLICY=passive runs at 35 GB/s and segfaulted. If llama.cpp
  serving with CPU experts shows mystery slowdowns, look here first.

## Architectural EP: kill the ~246 µs serial routing floor

The measured floor after --shared-late is x upload + routing compute +
sync + tiny reads. Two designs, both integration-scale:
- Replicated routing: every die holds the 11 MB router and computes top-k
  locally — zero round trip. Blocked on ggml having no on-device stream
  compaction for per-die pair selection (would need a custom kernel or a
  masked-compute trick that doesn't 4x the work).
- Router-fused per-die graphs (routing inside each die's graph, ids
  consumed on-device by mul_mat_id directly at full [6, n] shape with
  per-die expert masking).

## Upstream reports owed to llama.cpp

- `--spec-draft-n-max > 5` with the DSpark drafter does not clamp safely:
  the warning fires, then the first request dies with MUL_MAT failed /
  ROCm error (b10472, gfx906). Repro: chat.sh server + n_max 8.
- Compact-layout mul_mat_id cliff: ids [1, P] puts P in dst->ne[2] and
  falls off the MMVQ fast path past get_mmvq_mmid_max_batch — 9 pairs
  cost 3x what 8 do on gfx906/MXFP4. Maybe expected behavior, but the
  cliff shape is surprising; at minimum documentation-worthy.

## Serving-mode wait policy

The bench spins (ROCm blocking sync busy-waits by default); serving with
CPU expert layers wants blocking sync so the waiter doesn't compete with
the 16 expert threads. Measure decode t/s of the chat.sh config with
hipDeviceScheduleBlockingSync (env/patch) vs default — prediction: ~free
on HIP (poll-vs-block gap ~1 µs), frees a core.

## Housekeeping

- ~~perf spin verification~~ DONE 2026-08-18: 6 s / 4576 samples during
  the EP wait phase put ~55% of cycles in one ~48-byte loop in
  libhsa-runtime64 (the completion-signal spin-wait) vs 3.4% in the
  bench's own code — the pegged core polls ~16x more than it computes.
  (perf note: use /usr/lib/linux-tools/5.15.0-*/perf directly; the
  wrapper refuses the T2 kernel.)
- Windows-side SHA-256 cross-check of the DS-V4-Flash transfer against
  checksums/DS-V4-Flash-0731-UD-Q8_K_XL.sha256 (Linux-side manifest).
- Apple SSD (nvme0) health check from macOS someday — Linux no longer
  depends on it (boot migrated 2026-08-18), but macOS's ESP shares it.
- E6 occupancy experiment re-run under hipcc if the custom kernel ever
  matters again — the E1-E8 ledger was tuned under the Vulkan compiler
  (hipcc lands 35-36 VGPRs / 7 waves per SIMD vs Vulkan's 6).
