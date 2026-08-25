# dspark-tree — offline draft-tree study for DSpark speculative decoding

**Question.** hip-moe's online sweep (hip-moe/README.md, "DSpark speculative
decoding") measured single-*chain* drafting: n_max=3 optimum, 1.53x, acceptance
0.76. If drafting were a top-k *tree* instead, how often would the true
continuation be inside the tree — and how many draft tokens does a mean
accepted length of L cost? Nothing here implements tree decoding; the tree is
studied offline, exactly.

## Why the study is exact, not simulated

DSpark's drafter (`dflash` arch + Markov head) emits a whole block of 5 tokens
in one non-causal decode over `[anchor, MASK x4]`. Verified in
`llama.cpp/src/models/dflash.cpp` (`build_dspark_markov_head`) and
`common/speculative.cpp`:

- the **base logits** of every block slot depend only on the anchor and the
  draft KV state — *not* on which tokens are drafted;
- branching enters *only* through an additive rank-256 bias:
  `cond_i(prev) = base_i + markov_w2 @ markov_w1[prev]`
  (first-order Markov: slot i's distribution given any tree prefix depends
  only on the immediately preceding token).

So one draft call per position, plus the two 256x129280 Markov matrices read
from the drafter GGUF, reconstructs the drafter's **entire** top-k tree at
that position in numpy. No per-branch reruns, no approximation.

## Tools

- `src/tree_dump.cpp` (build: `make`, links against
  `~/projects/llama.cpp/build-hip` shared libs) — generates a greedy reference
  sequence with the target while feeding the drafter exactly as online
  speculation would (drafts are never committed, so the reference is the
  target's pure greedy continuation). Per position it dumps: the drafter's
  full-vocab **biased** logits for all 5 slots, the pre-Markov **base** logits
  (captured with a `cb_eval` callback on the `result_output` tensor), the
  5 per-slot confidences, and the drafter's own greedy chain.
  Prompts via `DSTREE_PROMPTS` (comma-separated files), output dir via
  `DSTREE_OUT`; all other flags are stock llama.cpp (`-m -md -ngl -ncmoe -ts
  --spec-type draft-dspark ...`). Format documented at the top of the source.
- `analyze.py` (venv: `.venv`, numpy + pyyaml; gguf-py imported from the
  llama.cpp checkout) — validates the reconstruction per record
  (`biased == base + bias(greedy prev)` elementwise), then computes, per
  position, the rank of the true token at each slot **conditioned on the true
  prefix**, and derives every tree metric from those ranks.

Run recipe (the validated hip-moe placement):

```
DSTREE_PROMPTS=prompts/python.txt,...,prompts/math.txt DSTREE_OUT=results \
./bin/tree-dump \
  -m ~/llms/DS-V4-Flash-0731-UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
  -md ~/llms/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  -ngl 99 -ncmoe 13 -ts 19/8/8/8 -t 16 -c 8192 --fit off \
  --spec-type draft-dspark -ngld 99 --spec-draft-cpu-moe -n 512
.venv/bin/python analyze.py results
```

## Metrics

- **Chain baseline** — mean accepted length of the greedy chain at n=1..5;
  cross-checks the online sweep's acceptance numbers on fixed text.
- **Top-k tree frontier** — per-depth width vectors (top-k_j children of every
  node at depth j): the true path survives depth d iff rank_j <= k_j for all
  j <= d; draft-token budget = sum_j prod(k_1..k_j). Best shape per budget.
- **Best-N-node tree** (EAGLE-2/SpecInfer style) — the N nodes with the
  highest joint drafter probability; containment counted exactly by
  best-first search. This is the near-optimal frontier.
- **Confidence calibration** — the drafter's in-graph confidence head vs
  empirical correctness along the greedy chain.

Depth caps at 5 (the drafter's trained `block_size`); tree acceptance beyond
5 would need a different drafter checkpoint, not a bigger tree.

## Results (6 prompts x 512 greedy tokens, 3048 full-depth positions, 2026-08-25)

Reconstruction sanity: max |biased - base - bias(prev)| = **4.6e-05** over all
15,360 slots; 0 chain/argmax mismatches in the dumps. Cross-check: offline
chain acceptance at n=3 is **0.762** vs the online sweep's **0.76** — the
offline study reproduces the online measurement on fixed text.

### Drafter quality per slot (conditioned on the true prefix)

| slot | top-1 | top-2 | top-4 | top-8 | top-16 |
|------|-------|-------|-------|-------|--------|
| 1 | 0.877 | 0.934 | 0.968 | 0.982 | 0.991 |
| 2 | 0.818 | 0.886 | 0.922 | 0.947 | 0.963 |
| 3 | 0.761 | 0.835 | 0.883 | 0.907 | 0.933 |
| 4 | 0.703 | 0.785 | 0.839 | 0.875 | 0.907 |
| 5 | 0.664 | 0.751 | 0.809 | 0.851 | 0.884 |

### Chain baseline, and the shape of acceptance

Mean accepted length: 0.88 / 1.63 / 2.29 / 2.84 / **3.34** at n = 1..5.

The n=5 mean of 3.34 is a fiction — the distribution is **U-shaped**, not
bell-shaped:

| accepted | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|-----|-----|-----|-----|-----|------|
| pooled | 12.3% | 12.1% | 10.5% | 9.1% | 7.1% | **48.9%** |
| c | 1.8 | 3.1 | 4.7 | 6.5 | 5.7 | **78.1** |
| python | 3.7 | 3.5 | 5.1 | 6.1 | 6.5 | **75.0** |
| js | 5.7 | 7.9 | 7.9 | 8.7 | 6.7 | **63.2** |
| math | 4.3 | 6.9 | 6.9 | 9.1 | 9.1 | **63.8** |
| prose | **28.7** | 27.4 | 19.7 | 11.0 | 6.5 | 6.7 |
| story | **29.3** | 23.8 | 18.5 | 13.4 | 8.3 | 6.7 |

The pooled bimodality decomposes by content: on code the drafter is "on
rails" (a full block accepted 75-78% of the time), on creative prose it is
"lost" (mode at 0-1). Acceptance is also nearly **memoryless**: P(slot
correct | prefix correct) = 0.877, 0.862, 0.861, 0.860, 0.873 across slots
1..5 — the drafter does not degrade with depth, it either tracks the target
or it doesn't. Consequence: **block_size = 5 truncates a fat right tail** —
half of all drafts (three quarters on code) would extend past 5 at a ~0.86
per-slot continuation rate. Depth, not width, is where the unclaimed
tokens are.

### Top-k tree frontier (best per-depth width vector per budget)

| draft tokens | shape | mean accepted | tokens/verify |
|--------------|-----------|-------|-------|
| 5 (chain) | 1x1x1x1x1 | 3.335 | 4.335 |
| 10 | 2x1x1x1x1 | 3.454 | 4.454 |
| 18 | 2x2x1x1x1 | 3.580 | 4.580 |
| 30 | 2x2x2x1x1 | 3.678 | 4.678 |
| 46 | 2x2x2x2x1 | 3.763 | 4.763 |
| 62 | 2x2x2x2x2 | 3.809 | 4.809 |

The best 62-token tree buys **+0.47** accepted tokens over the 5-token chain
— 12x the verify batch for a 14% gain. Depth is capped at 5, so width can
only rescue positions where the true token is ranked 2+, and per-slot top-1
is already 0.87/0.66 (slot 1/5). Trees help hard content relatively more
(prose chain-3 1.39 -> 1.80 with an 8-token 2x2x1 tree, +29%; python +4%),
but hard content is exactly where absolute acceptance stays low.

### Best-N-node tree (EAGLE-2-style, nodes ranked by joint drafter prob)

The near-optimal frontier: the tree containing the N nodes with the highest
joint drafter probability (containment counted exactly by best-first search;
at N <= 5 it coincides with the chain, as it must — a consistency check the
first version of the counting failed by ~1e-7 of float32 rounding).

| N (draft tokens) | mean accepted | tokens/verify |
|-----|-------|-------|
| 5 (= chain) | 3.34 | 4.34 |
| 6 | 3.51 | 4.51 |
| 8 | 3.65 | 4.65 |
| 12 | 3.79 | 4.79 |
| 16 | 3.87 | 4.87 |
| 24 | 3.95 | 4.95 |
| 32 | 4.01 | 5.01 |
| 48 | 4.07 | 5.07 |
| 64 | 4.11 | 5.11 |
| 96 | 4.16 | 5.16 |
| 128 | 4.21 | 5.21 |

At equal budget the adaptive tree beats every fixed top-k shape (16 nodes:
3.87 vs 3.58 for the 18-token 2x2x1x1x1; 64: 4.11 vs 3.81 for the 62-token
2x2x2x2x2) — but the returns are brutally logarithmic past ~8 nodes:
**+0.7 accepted tokens costs 25x the draft budget** (5 -> 128).

### The headline answer: tree size for mean accepted length L

| target mean accepted L | draft tokens needed (best tree) |
|---|---|
| 3.3 | 5 (the chain) |
| 3.5 | 6 |
| 3.9 | 16 |
| 4.0 | 32 |
| 4.2 | 128 |
| 4.3+ | out of reach at any width (depth-5 ceiling) |

Even one extra mean accepted token over the n=5 chain is unreachable within
128 nodes. On this box the verdict is stronger still: hip-moe measured verify
cost rising steeply with batch (n=3 chain optimal online, 13 expert layers on
CPU), so a 16-32 token verify batch would cost far more than the ~18%
acceptance gain buys. Width is not where the tokens are; depth is (see the
truncated-tail argument above).

### Confidence head calibration (greedy chain, slots pooled)

| conf bucket | mean conf | P(correct) | n |
|-------------|-----------|------------|------|
| [0.00,0.20) | 0.164 | 0.059 | 747 |
| [0.20,0.40) | 0.298 | 0.189 | 2009 |
| [0.40,0.50) | 0.451 | 0.316 | 882 |
| [0.50,0.60) | 0.551 | 0.466 | 948 |
| [0.60,0.70) | 0.651 | 0.587 | 888 |
| [0.70,0.80) | 0.751 | 0.698 | 998 |
| [0.80,0.90) | 0.850 | 0.816 | 1224 |
| [0.90,0.95) | 0.927 | 0.887 | 758 |
| [0.95,1.00) | 0.994 | 0.985 | 6763 |

Well calibrated (mild ~5-point overconfidence), excellent at the top — and
two thirds of all slots sit in the top bucket. `--spec-draft-p-min` is
therefore a trustworthy knob, and a confidence-gated *variable-length* draft
is well supported by the model itself.

### Is depth 5 a model limit or a platform limit?

A **checkpoint** limit, not an architecture or llama.cpp limit:

- `dflash.block_size = 5` is GGUF metadata written at conversion — this
  SpecForge export was *trained* to denoise 5-token blocks. llama.cpp reads
  the key and clamps `--spec-draft-n-max` to it (`common/speculative.cpp`,
  warn-and-clamp); the Markov head refuses blocks longer than trained
  (`dflash.cpp`, `block_drafts > block_size -> return`). The code's fallback
  default is `block_size = 16`, i.e. the platform expects other checkpoints
  to declare bigger blocks.
- It cannot be worked around at inference time: the drafter conditions on
  *target-model hidden states* up to the anchor. Drafted tokens have no
  target features until they are verified, so a second draft call cannot
  extend the first — depth beyond block_size requires a drafter retrained
  (or re-exported) with a larger block.
- Given the memoryless ~0.86 continuation rate and the truncated right tail
  above, a deeper-block DSpark checkpoint is the single most promising lever
  this study surfaced.

## Notes / caveats

- Corpus: 6 prompts (python, js, c, technical prose, short story, math proof)
  recreating the shape of the online sweep's uncommitted corpus; raw
  completion (no chat template), greedy.
- The confidence head's branch-conditional value is not recoverable offline
  (the graph overwrites the hidden state it reads with the broadcast
  confidences), so calibration is reported along the greedy chain only.
- Reconstruction tolerance: bf16 Markov weights dequantize losslessly;
  observed max elementwise error ~1e-5 (fp32 accumulation order), far below
  logit gaps that decide ranks.
