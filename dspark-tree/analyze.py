#!/usr/bin/env python3
"""Offline DSpark draft-tree analysis.

Reads tree-dump's .dstree files (per-position full-vocab drafter logits, both
biased and pre-Markov base) plus the drafter GGUF's Markov tensors, and
answers: if drafting were a top-k tree instead of a single chain, how large
must the tree be for a mean accepted length of L?

Reconstruction identity (validated per record against the dumped tensors):
    cond_i(prev) = base_i + markov_w2 @ markov_w1[prev]
where base_i is branch-independent. Conditioning is first-order Markov, so
the drafter's distribution at slot i given ANY tree prefix depends only on the
immediately preceding token — the whole tree follows from one draft call.

Stages:
  1. sanity   — elementwise check biased == base + bias(greedy prev)
  2. stats    — per-record: rank & logprob of the true token at each slot,
                conditioned on the TRUE prefix; confidences; chain match
  3. report   — chain acceptance, top-k-tree frontier, best-N-node frontier,
                per-prompt breakdown, confidence calibration

Usage:
  .venv/bin/python analyze.py results/ [--gguf PATH] [--sanity-only]
                              [--no-best-n] [--best-n-max 128]
"""

import argparse
import functools
import json
import sys
from pathlib import Path

# progress must be visible while the run is alive, also when stdout is a file
print = functools.partial(print, flush=True)

import numpy as np

sys.path.insert(0, str(Path.home() / "projects/llama.cpp/gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

DEFAULT_GGUF = Path.home() / "llms/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"

MAGIC = b"DSPKTRE2"


def load_markov(gguf_path):
    r = gguf.GGUFReader(str(gguf_path))
    tensors = {t.name: t for t in r.tensors}
    w1 = dequantize(tensors["markov_w1.weight"].data, tensors["markov_w1.weight"].tensor_type)
    w2 = dequantize(tensors["markov_w2.weight"].data, tensors["markov_w2.weight"].tensor_type)
    # gguf-py returns token-major (n_vocab, rank); bias(prev) = w2 @ w1[prev]
    assert w1.shape == w2.shape and w1.shape[1] < w1.shape[0], w1.shape
    return w1.astype(np.float32), w2.astype(np.float32)


def open_dump(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
    assert hdr[:8] == MAGIC, f"{path}: bad magic {hdr[:8]!r}"
    n_vocab, block = np.frombuffer(hdr[8:], dtype="<u4")
    rec_dtype = np.dtype([
        ("step",   "<u4"),
        ("n_past", "<u4"),
        ("anchor", "<i4"),
        ("chain",  "<i4", (block,)),
        ("conf",   "<f4", (block,)),
        ("logits", "<f4", (block, n_vocab)),
        ("base",   "<f4", (block, n_vocab)),
    ])
    size = Path(path).stat().st_size - 16
    assert size % rec_dtype.itemsize == 0, f"{path}: truncated file"
    recs = np.memmap(path, dtype=rec_dtype, mode="r", offset=16)
    return int(n_vocab), int(block), recs


def bias_for(w1, w2, prevs):
    """bias rows for a list of prev tokens -> (len(prevs), n_vocab)."""
    return w1[np.asarray(prevs)] @ w2.T


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return x - m - s


def process_file(dump_path, meta_path, w1, w2, sanity_only, best_n_max, quiet=False):
    n_vocab, B, recs = open_dump(dump_path)
    if not quiet:
        print(f"  processing {Path(dump_path).name} ({len(recs)} records) ...")
    meta = json.loads(Path(meta_path).read_text())
    tokens = np.asarray(meta["tokens"], dtype=np.int64)

    max_err = 0.0
    n_slots_checked = 0

    ranks = []       # (n_used, B) rank of the true token in cond (1 = top)
    logps = []       # (n_used, B) logprob of the true token in cond
    confs = []       # (n_used, B)
    chain_ok = []    # (n_used, B) drafter greedy chain token == true token
    depth_avail = [] # usable depth per record
    nstars = []      # (n_used, B) best-N: #nodes with joint logp >= true path's

    for rec in recs:
        n_past = int(rec["n_past"])
        true = tokens[n_past + 1 : n_past + 1 + B]
        d = len(true)
        if d == 0:
            continue

        base   = np.asarray(rec["base"],   dtype=np.float32)
        biased = np.asarray(rec["logits"], dtype=np.float32)
        chain  = np.asarray(rec["chain"])

        # ---- sanity: biased[i] == base[i] + bias(greedy prev_i) ----
        prevs_greedy = np.concatenate(([rec["anchor"]], chain[:-1]))
        recon = base + bias_for(w1, w2, prevs_greedy)
        err = float(np.max(np.abs(recon - biased)))
        max_err = max(max_err, err)
        n_slots_checked += B
        if sanity_only:
            continue

        # ---- true-prefix conditionals ----
        prevs_true = np.concatenate(([rec["anchor"]], true[:-1]))
        cond = base[:d] + bias_for(w1, w2, prevs_true[:d])
        lp = log_softmax(cond, axis=1)

        r_rec  = np.full(B, np.iinfo(np.int32).max, dtype=np.int64)
        lp_rec = np.full(B, np.nan, dtype=np.float32)
        for i in range(d):
            t = true[i]
            r_rec[i]  = int(np.sum(cond[i] > cond[i, t])) + 1
            lp_rec[i] = lp[i, t]

        ranks.append(r_rec)
        logps.append(lp_rec)
        confs.append(np.asarray(rec["conf"], dtype=np.float32))
        chain_ok.append(np.concatenate((chain[:d] == true, np.zeros(B - d, dtype=bool))))
        depth_avail.append(d)

        # ---- best-N-node tree: n_star_d = #nodes with joint logp >= q_d ----
        if best_n_max > 0:
            nstars.append(best_n_star(base, w1, w2, prevs_true, true, d, best_n_max, B))

    stats = {
        "name": Path(dump_path).stem,
        "n_records": len(depth_avail) if not sanity_only else len(recs),
        "max_err": max_err,
        "n_slots_checked": n_slots_checked,
        "n_chain_mismatch_dump": meta.get("n_chain_mismatch", -1),
    }
    if not sanity_only:
        stats.update({
            "ranks":  np.array(ranks),
            "logps":  np.array(logps),
            "confs":  np.array(confs),
            "chain_ok": np.array(chain_ok),
            "depth":  np.array(depth_avail),
            "nstars": np.array(nstars) if nstars else None,
        })
    if not quiet:
        print(f"  {stats['name']}: {stats['n_records']} records, "
              f"reconstruction max|err| = {max_err:.2e}")
    return stats


def best_n_star(base, w1, w2, prevs_true, true, d, n_max, B):
    """n_star_d = size the best-N-node tree must reach for the true path to be
    contained to depth d. The tree is the top-N nodes by joint drafter prob
    (EAGLE-2 style), ties broken in the true path's favor (consistent with the
    strict-> rank definition):

        n_star_d = max(d, 1 + #{nodes with joint logp > q_d})

    where q_d is the true path's joint logp. The max(d, .) floor covers exact
    ties with the path's own ancestors (a tree cannot contain a node without
    its ancestors). Best-first search, exact up to n_max (values whose count
    could not be completed within n_max pops are reported as n_max + 1).

    Precision: q_d is computed with the SAME expression chain as the heap
    children (float32 log_softmax + float64 running total), so the true node's
    heap value equals q_d bit-for-bit. A float64 accumulation here once made
    near-certain continuations (conditional prob ~ 1) land ~1e-7 away from
    their heap twins, undercounting nodes and yielding impossible rows like
    n_star = [1, 1, ...].
    """
    import heapq

    lp_true_path = np.empty(d)
    lp_node = 0.0
    for i in range(d):
        cond = base[i] + w1[prevs_true[i]] @ w2.T
        lp_c = log_softmax(cond) + lp_node   # float32 + float64 scalar, as in the search
        lp_node = float(lp_c[true[i]])
        lp_true_path[i] = lp_node
    q_min = lp_true_path[-1]

    # root children (depth 1)
    cond = base[0] + w1[prevs_true[0]] @ w2.T
    lp1 = log_softmax(cond) + 0.0
    # heap of (-joint_lp, depth, token); expand best-first
    cand = np.flatnonzero(lp1 >= q_min)
    heap = [(-float(lp1[t]), 1, int(t)) for t in cand]
    heapq.heapify(heap)

    popped = 0
    last_lp = np.inf
    greater = np.zeros(B, dtype=np.int64)  # nodes (any depth) with lp > q_d, strictly
    while heap and popped < n_max:
        neg_lp, depth, tok = heapq.heappop(heap)
        lp_node = -neg_lp
        last_lp = lp_node
        popped += 1
        for i in range(d):
            if lp_node > lp_true_path[i]:
                greater[i] += 1
        if depth < d:
            cond = base[depth] + w1[tok] @ w2.T
            lp_c = log_softmax(cond) + lp_node
            cand = np.flatnonzero(lp_c >= q_min)
            for t in cand:
                heapq.heappush(heap, (-float(lp_c[t]), depth + 1, int(t)))

    exhausted = not heap
    out = np.full(B, n_max + 1, dtype=np.int64)
    for i in range(d):
        # the strictly-greater count is complete iff best-first got past q_d
        if exhausted or last_lp <= lp_true_path[i]:
            out[i] = min(max(i + 1, 1 + greater[i]), n_max + 1)
    return out


# ---------------- reporting ----------------

def accepted_len_dist(ranks):
    """Distribution of chain accepted length (leading run of rank-1 slots)."""
    B = ranks.shape[1]
    ok = ranks == 1
    acc = np.zeros(len(ranks), dtype=np.int64)
    alive = np.ones(len(ranks), dtype=bool)
    for i in range(B):
        alive &= ok[:, i]
        acc[alive] += 1
    return np.bincount(acc, minlength=B + 1)


def accepted_len_topk(ranks, ks):
    """Mean accepted length for a per-depth-top-k tree with widths ks."""
    ok = np.ones(len(ranks), dtype=bool)
    total = np.zeros(len(ranks), dtype=np.float64)
    for i, k in enumerate(ks):
        ok = ok & (ranks[:, i] <= k)
        total += ok
    return float(np.mean(total))


def tree_size(ks):
    """Number of draft tokens in a per-depth-top-k tree."""
    total, layer = 0, 1
    for k in ks:
        layer *= k
        total += layer
    return total


def enumerate_shapes(max_tokens, B):
    """Non-increasing width vectors with size <= max_tokens."""
    shapes = []

    def rec(prefix, budget):
        if prefix:
            shapes.append(tuple(prefix))
        if len(prefix) == B:
            return
        hi = prefix[-1] if prefix else max_tokens
        for k in range(1, hi + 1):
            if tree_size(prefix + [k]) > budget:
                break
            rec(prefix + [k], budget)

    rec([], max_tokens)
    return shapes


def report(all_stats, best_n_max):
    ranks  = np.concatenate([s["ranks"] for s in all_stats])
    depth  = np.concatenate([s["depth"] for s in all_stats])
    B = ranks.shape[1]

    full = depth == B          # records with a full-depth true continuation
    ranks_f = ranks[full]
    n = len(ranks_f)
    print(f"\n=== {n} full-depth records (of {len(ranks)}) across {len(all_stats)} prompts ===")

    # --- per-slot marginals, conditioned on the true prefix ---
    print("\nP(true token in drafter's top-k at slot i | true prefix):")
    print("  slot |   top-1    top-2    top-4    top-8   top-16")
    for i in range(B):
        row = [np.mean(ranks_f[:, i] <= k) for k in (1, 2, 4, 8, 16)]
        print(f"     {i+1} | " + "  ".join(f"{v:7.3f}" for v in row))

    # --- chain baseline ---
    print("\nSingle-chain (greedy) mean accepted length:")
    for nmax in range(1, B + 1):
        m = accepted_len_topk(ranks_f, [1] * nmax)
        print(f"  n={nmax}: {m:.3f} accepted of {nmax} drafted "
              f"({m/nmax:.3f} acceptance, {1+m:.3f} tokens/verify)")

    # --- accepted-length distribution at full depth ---
    print(f"\nChain n={B} accepted-length distribution (leading rank-1 run):")
    h = accepted_len_dist(ranks_f)
    for k in range(B + 1):
        print(f"  {k}: {h[k]/n*100:5.1f}%  {'#' * int(h[k]/n*60)}")
    surv = np.cumsum(h[::-1])[::-1]  # P(accepted >= k) * n
    cont = [surv[k+1]/surv[k] for k in range(B) if surv[k] > 0]
    print("  P(slot correct | prefix correct) per slot: "
          + "  ".join(f"{v:.3f}" for v in cont))
    print("  per prompt (% at each accepted length 0..{}):".format(B))
    for s in all_stats:
        rf = s["ranks"][s["depth"] == B]
        if len(rf) == 0:
            continue
        hp = accepted_len_dist(rf) / len(rf) * 100
        print(f"    {s['name']:>8} | " + " ".join(f"{v:5.1f}" for v in hp))

    # --- per-depth-top-k frontier ---
    print("\nTop-k tree frontier (best non-increasing width vector per budget):")
    shapes = enumerate_shapes(64, B)
    best = {}
    for ks in shapes:
        sz = tree_size(ks)
        m = accepted_len_topk(ranks_f, list(ks))
        if sz not in best or m > best[sz][0]:
            best[sz] = (m, ks)
    print("  draft-tokens | mean accepted | tokens/verify | shape")
    prev_m = -1.0
    for sz in sorted(best):
        m, ks = best[sz]
        if m <= prev_m + 1e-9:
            continue
        prev_m = m
        print(f"       {sz:5d} |       {m:6.3f} |        {1+m:6.3f} | {'x'.join(map(str, ks))}")

    # --- best-N-node tree ---
    if all(s["nstars"] is not None for s in all_stats):
        nstars = np.concatenate([s["nstars"] for s in all_stats])[full]
        print(f"\nBest-N-node tree (nodes ranked by joint drafter prob, exact up to N={best_n_max}):")
        print("  N (draft-tokens) | mean accepted | tokens/verify")
        for N in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
            if N > best_n_max:
                break
            contained = nstars <= N          # (n, B) cumulative by construction
            m = float(np.mean(np.sum(contained, axis=1)))
            print(f"          {N:6d} |       {m:6.3f} |        {1+m:6.3f}")

    # --- per-prompt breakdown ---
    print("\nPer-prompt mean accepted length (chain n=3 / n=5 | top-k tree 2x2x1 (8 tok) / 4x2x2x1x1 (30 tok)):")
    for s in all_stats:
        rf = s["ranks"][s["depth"] == B]
        if len(rf) == 0:
            continue
        c3 = accepted_len_topk(rf, [1, 1, 1])
        c5 = accepted_len_topk(rf, [1] * B)
        t8 = accepted_len_topk(rf, [2, 2, 1])
        t30 = accepted_len_topk(rf, [4, 2, 2, 1, 1])
        print(f"  {s['name']:>8}: {c3:.3f} / {c5:.3f} | {t8:.3f} / {t30:.3f}   ({len(rf)} records)")

    # --- confidence calibration (greedy path) ---
    confs = np.concatenate([s["confs"] for s in all_stats])[full]
    chain_ok = np.concatenate([s["chain_ok"] for s in all_stats])[full]
    print("\nConfidence-head calibration (greedy chain, all slots pooled):")
    print("  conf bucket | mean conf | P(chain token correct) | count")
    edges = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    c = confs.ravel()
    ok = chain_ok.ravel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (c >= lo) & (c < hi)
        if np.sum(sel) == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f}) |    {np.mean(c[sel]):.3f}  |          {np.mean(ok[sel]):.3f}       | {np.sum(sel)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--no-best-n", action="store_true")
    ap.add_argument("--best-n-max", type=int, default=128)
    ap.add_argument("--save", type=Path, help="save per-record stats to this .npz")
    args = ap.parse_args()

    print(f"loading markov tensors from {args.gguf} ...")
    w1, w2 = load_markov(args.gguf)

    dumps = sorted(args.dump_dir.glob("*.dstree"))
    if not dumps:
        sys.exit(f"no .dstree files in {args.dump_dir}")

    best_n_max = 0 if (args.no_best_n or args.sanity_only) else args.best_n_max

    all_stats = []
    for dp in dumps:
        mp = dp.with_suffix("").with_suffix("")  # strip .dstree
        mp = dp.parent / (dp.stem + ".meta.json")
        all_stats.append(process_file(dp, mp, w1, w2, args.sanity_only, best_n_max))

    worst = max(s["max_err"] for s in all_stats)
    total_slots = sum(s["n_slots_checked"] for s in all_stats)
    print(f"\nsanity: reconstruction max|biased - base - bias(prev)| = {worst:.2e} "
          f"over {total_slots} slots")
    if worst > 1e-2:
        sys.exit("RECONSTRUCTION FAILED - do not trust any numbers above this line")

    if args.sanity_only:
        return

    if args.save:
        np.savez_compressed(args.save, **{
            f"{s['name']}/{k}": v for s in all_stats
            for k, v in s.items() if isinstance(v, np.ndarray)
        })
        print(f"saved per-record stats to {args.save}")

    report(all_stats, best_n_max)


if __name__ == "__main__":
    main()
