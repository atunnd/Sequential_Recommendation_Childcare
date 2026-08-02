#!/usr/bin/env python3
"""
Inference: produce a childcare recommendation from cached artifacts, without retraining.

    python scripts/infer.py --n 5                          # first 5 test respondents
    python scripts/infer.py --n 5 --random                 # random sample of the test split
    python scripts/infer.py --tucaseid 20030100013280      # one known respondent
    python scripts/infer.py --diary my_day.json            # an unseen diary (b1/b3/b4)
    python scripts/infer.py --baseline b3 --n 20 --out recs.json --quiet

What it reuses vs. what it recomputes
-------------------------------------
Reused from artifacts/: the encoder (b3/b4 checkpoints), the fitted clusterer, and the
cached population embeddings. Nothing is refit.

Recomputed every run: the exemplar pool. `find_exemplars` ranks peers by childcare minutes
normalized *within their own cluster*, so it needs every respondent's diary, not just the
query's. That means the raw CSVs still have to be read. To keep repeat calls cheap this
script caches the parsed sequences in artifacts/sequences_cache.pkl, keyed on the size and
mtime of both CSVs, so only the first run pays the ~524M parse. Pass --no-cache to skip it.

The pool is rebuilt exactly as run_pipeline.py builds it — train + test embeddings, val
excluded — so a query already in the dataset reproduces the recommendation in
<baseline>_results.json byte for byte.

Out-of-sample diaries (--diary)
-------------------------------
Supported for b1 (stateless), b3 and b4 (checkpoints are saved). NOT supported for b2:
B2Pipeline fits a TruncatedSVD that is never persisted to disk, only its output embeddings
are, so an unseen diary cannot be projected into B2 space without refitting.

Diary JSON format — durations in minutes, parallel to activities:

    {"activities": [0, 2, 14, 4, 7], "durations": [480, 60, 120, 45, 480]}

or with raw 6-digit ATUS codes instead of category ids (mapped via activity_mapping.yaml):

    {"raw_codes": ["010101", "020201"], "durations": [480, 60]}
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_activity_mapping
from src.preprocessing.loader import load_raw, map_activities
from src.preprocessing.sequences import build_sequences, CHILDCARE_CAT
from src.recommendation.health_score import compute_childcare_scores
from src.recommendation.exemplars import find_exemplars
from src.recommendation.generator import generate_recommendation

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
SEQ_CACHE = ARTIFACTS / "sequences_cache.pkl"
SLEEP_CAT = 0


# artifact loading 
def clusterer_path(baseline: str, clustering: str) -> Path:
    return ARTIFACTS / ("bgmm_model.pkl" if baseline == "b4" else f"{baseline}_bgmm_model.pkl")


def require(path: Path, hint: str):
    if not path.exists():
        sys.exit(f"Missing {path}\n  {hint}")
    return path


def load_sequences(use_cache: bool = True) -> dict:
    """Parse the raw CSVs into per-respondent sequences, memoized on the CSVs' stat."""
    from src.config import ATUSACT_CSV, ATUSSUM_CSV

    def stamp():
        return [
            [str(p), p.stat().st_size, int(p.stat().st_mtime)]
            for p in (ATUSACT_CSV, ATUSSUM_CSV)
        ]

    if use_cache and SEQ_CACHE.exists():
        try:
            with open(SEQ_CACHE, "rb") as f:
                blob = pickle.load(f)
            if blob.get("stamp") == stamp():
                print(f"Sequences from cache ({len(blob['sequences']):,} respondents)")
                return blob["sequences"]
            print("Sequence cache is stale (raw CSVs changed) — reparsing.")
        except Exception as exc:                      # corrupt/partial cache is not fatal
            print(f"Sequence cache unreadable ({exc}) — reparsing.")

    print("Reading raw ATUS data (slow, ~524M)...")
    sequences = build_sequences(map_activities(load_raw()))

    if use_cache:
        try:
            with open(SEQ_CACHE, "wb") as f:
                pickle.dump({"stamp": stamp(), "sequences": sequences}, f)
            print(f"Cached sequences -> {SEQ_CACHE.relative_to(ARTIFACTS.parent)}")
        except Exception as exc:
            print(f"Could not write sequence cache ({exc}) — continuing.")
    return sequences


def build_pool(baseline: str, clustering: str):
    """
    Rebuild the exemplar pool exactly as run_pipeline.py does: train + test embeddings
    (val is excluded there too), labelled by the fitted clusterer.
    """
    emb_path = require(
        ARTIFACTS / f"{baseline}_embeddings.pkl",
        f"Train it first:  ./scripts/train.sh {baseline}",
    )
    clu_path = require(
        clusterer_path(baseline, clustering),
        f"Train it first:  CLUSTERING={clustering} ./scripts/train.sh {baseline}",
    )

    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    with open(clu_path, "rb") as f:
        clusterer = pickle.load(f)

    train_ids, X_train = emb["train"]
    test_ids, X_test = emb["test"]

    all_ids = list(train_ids) + list(test_ids)
    all_X = np.vstack([X_train, X_test])
    all_labels = np.concatenate([clusterer.predict(X_train), clusterer.predict(X_test)])

    print(f"Pool: {len(all_ids):,} respondents, K={clusterer.k}, dim={all_X.shape[1]}")
    return clusterer, all_ids, all_labels, list(test_ids)


def sleep_median_of(sequences: dict, ids: list) -> float:
    """Population sleep median — the feasibility floor used by find_exemplars."""
    totals = [
        sum(d for a, d in zip(sequences[i]["activities"], sequences[i]["durations"])
            if a == SLEEP_CAT)
        for i in ids
    ]
    return float(np.median(totals))


# unseen diaries 
def read_diary(path: Path) -> dict:
    """Turn a diary JSON into the same dict shape build_sequences() emits."""
    raw = json.loads(Path(path).read_text())

    durations = [int(d) for d in raw["durations"]]
    if "activities" in raw:
        activities = [int(a) for a in raw["activities"]]
        raw_codes = [str(c) for c in raw.get("raw_codes", [])] or ["000000"] * len(activities)
    elif "raw_codes" in raw:
        mapping = load_activity_mapping()
        raw_codes = [str(c).zfill(6) for c in raw["raw_codes"]]
        activities = [int(mapping.get(c, 18)) for c in raw_codes]   # 18 = Other/Unknown
    else:
        sys.exit(f"{path}: need an 'activities' or a 'raw_codes' field.")

    if len(activities) != len(durations):
        sys.exit(f"{path}: {len(activities)} activities but {len(durations)} durations.")
    if not activities:
        sys.exit(f"{path}: empty diary.")
    bad = [a for a in activities if not 0 <= a <= 18]
    if bad:
        sys.exit(f"{path}: category ids out of range 0-18: {sorted(set(bad))}")

    total = sum(durations)
    if abs(total - 1440) > 60:
        print(f"  warning: diary totals {total} min, not ~1440 — ATUS days are full days.")

    return {
        "activities": activities,
        "durations": durations,
        "raw_codes": raw_codes,
        "start_times": [],
        "childcare_min": sum(d for a, d in zip(activities, durations) if a == CHILDCARE_CAT),
        "trchildnum": int(raw.get("trchildnum", 1)),
        "year": int(raw.get("year", -1)),
    }


def embed_unseen(baseline: str, seq: dict, device: str) -> np.ndarray:
    """Embed a single unseen diary with the persisted encoder."""
    holder = {"__query__": seq}

    if baseline == "b1":
        from src.baselines.b1_duration import compute_b1
        return compute_b1(seq["activities"], seq["durations"]).reshape(1, -1)

    if baseline == "b2":
        sys.exit(
            "b2 cannot embed an unseen diary: B2Pipeline's TruncatedSVD is fitted at train\n"
            "time and never saved (only b2_embeddings.pkl is). Use --baseline b1, b3 or b4,\n"
            "or query a tucaseid that is already in the dataset."
        )

    if baseline == "b3":
        from src.baselines.b3_transformer import B3Pipeline
        ckpt = require(ARTIFACTS / "b3_model.pt", "Train it first:  ./scripts/train.sh b3")
        _, X = B3Pipeline().load(str(ckpt), device=device).transform(holder, device=device)
        return X

    from src.baselines.b4_hybrid import B4Pipeline
    ckpt = require(ARTIFACTS / "b4_model.pt", "Train it first:  ./scripts/train.sh b4")
    _, X = B4Pipeline().load(str(ckpt), device=device).transform(holder, device=device)
    return X


# reporting
def print_recommendation(rec: dict, cluster_id: int, seq: dict):
    print("\n" + "─" * 76)
    print(f"Respondent {rec['query_id']}   cluster {cluster_id}   "
          f"{len(seq['activities'])} activities   {seq.get('trchildnum', '?')} child(ren)")
    print("─" * 76)

    if rec["level1"] is None:
        print("  No feasible recommendation: no source activity in this day has a positive")
        print("  transition gap toward Childcare relative to the exemplar pool.")
        return

    l1 = rec["level1"]
    print(f"  After {l1['source']}, do {l1['target']} instead of {l1['displaced']}")
    if rec["level2"]:
        l2 = rec["level2"]
        print(f"    specifically: {l2['activity_name']}  "
              f"(code {l2['code']}, {l2['frequency']:.0%} of exemplar transitions)")

    before, after = rec["original_childcare_min"], rec["new_childcare_min"]
    gap = rec["context"]["childcare_gap_minutes"]
    closure = min(max((after - before) / gap, 0.0), 1.0) if gap > 0 else float("nan")
    print(f"\n  Childcare   {before} -> {after} min  (+{after - before})")
    print(f"  Peer gap    {gap} min across {rec['exemplar_count']} exemplars"
          + (f"   closure {closure:.0%}" if gap > 0 else "   (already at/above peers)"))
    print(f"  Edits       {rec['edit_distance']} slot(s) changed")

    # Show only the slots that actually moved, not the whole 20-30 step day.
    old = rec["original_sequence"].split(" -> ")
    new = rec["new_sequence"].split(" -> ")
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    if changed:
        print("\n  Changed slots:")
        for i in changed:
            mins = seq["durations"][i] if i < len(seq["durations"]) else "?"
            prev = old[i - 1] if i > 0 else "(start of day)"
            print(f"    slot {i:>2} after {prev:<24} {old[i]} -> {new[i]}  ({mins} min)")


def print_table(rows: list):
    print(f"\n  {'tucaseid':>16} {'clu':>4} {'before':>7} {'after':>6} {'delta':>6} "
          f"{'edits':>6}  recommendation")
    for r in rows:
        rec = r["recommendation"]
        l1 = rec["level1"]
        what = (f"{l1['source']} -> Childcare (was {l1['displaced']})"
                if l1 else "(none feasible)")
        print(f"  {str(rec['query_id']):>16} {r['cluster_id']:>4} "
              f"{rec['original_childcare_min']:>7} {rec['new_childcare_min']:>6} "
              f"{rec['new_childcare_min'] - rec['original_childcare_min']:>+6} "
              f"{rec['edit_distance']:>6}  {what}")


# main
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate childcare recommendations from cached artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], default="b4")
    p.add_argument("--clustering", choices=["bgmm", "kmeans-elbow"], default="bgmm")
    p.add_argument("--tucaseid", nargs="+", help="one or more respondent ids to explain")
    p.add_argument("--n", type=int, help="number of test-split respondents to run")
    p.add_argument("--random", action="store_true", help="with --n, sample instead of head")
    p.add_argument("--seed", type=int, default=42, help="sampling seed for --random")
    p.add_argument("--diary", type=Path, help="JSON file holding an unseen diary")
    p.add_argument("--out", type=Path, help="write the recommendations to this JSON file")
    p.add_argument("--quiet", action="store_true", help="table only, no per-respondent detail")
    p.add_argument("--no-cache", action="store_true", help="do not use artifacts/sequences_cache.pkl")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if sum(x is not None and x is not False for x in (args.tucaseid, args.n, args.diary)) != 1:
        p.error("choose exactly one of --tucaseid, --n, --diary")
    return args


def main():
    args = parse_args()

    sequences = load_sequences(use_cache=not args.no_cache)
    clusterer, all_ids, all_labels, test_ids = build_pool(args.baseline, args.clustering)

    # tucaseid keys arrive from pandas as numpy int64; accept them as strings on the CLI.
    by_str = {str(k): k for k in all_ids}

    # choose the queries 
    if args.diary:
        seq = read_diary(args.diary)
        query_id = f"DIARY:{args.diary.stem}"
        vec = embed_unseen(args.baseline, seq, args.device)
        label = int(clusterer.predict(vec)[0])
        print(f"Unseen diary embedded with {args.baseline.upper()} -> cluster {label}")

        # Add the newcomer to the pool before scoring: within-cluster min-max normalization
        # is defined over the pool, so the query must be inside it.
        sequences = {**sequences, query_id: seq}
        all_ids = all_ids + [query_id]
        all_labels = np.concatenate([all_labels, [label]])
        queries = [query_id]

    elif args.tucaseid:
        queries = []
        for raw in args.tucaseid:
            if raw not in by_str:
                sys.exit(
                    f"tucaseid {raw} is not in the {args.baseline} train+test pool.\n"
                    "  It may be a non-parent (filtered out), or in the val split, which\n"
                    "  run_pipeline.py excludes from the pool. Use --diary for a new day."
                )
            queries.append(by_str[raw])

    else:
        n = min(args.n, len(test_ids))
        if args.random:
            rng = np.random.default_rng(args.seed)
            queries = [test_ids[i] for i in rng.choice(len(test_ids), n, replace=False)]
        else:
            queries = test_ids[:n]     # same order run_pipeline.py evaluates in

    # score and recommend
    childcare_scores = compute_childcare_scores(sequences, all_ids, all_labels)
    sleep_median = sleep_median_of(sequences, all_ids)
    id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
    print(f"Population sleep median: {sleep_median:.0f} min")

    results = []
    for uid in queries:
        cluster_id = int(all_labels[id_to_idx[uid]])
        exemplars = find_exemplars(
            query_id=uid,
            query_label=cluster_id,
            sequences=sequences,
            all_ids=all_ids,
            labels=all_labels,
            childcare_scores=childcare_scores,
            sleep_threshold=sleep_median,
        )
        if not exemplars:
            # Same fallback run_pipeline.py uses: nearest few cluster-mates, unfiltered.
            exemplars = [all_ids[i] for i in np.where(all_labels == cluster_id)[0][:5]
                         if all_ids[i] != uid]
        rec = generate_recommendation(uid, exemplars, sequences)
        results.append({
            "query_id": str(uid),
            "cluster_id": cluster_id,
            "exemplar_ids": [str(e) for e in exemplars[:5]],
            "recommendation": rec,
        })
        if not args.quiet:
            print_recommendation(rec, cluster_id, sequences[uid])

    if args.quiet or len(results) > 1:
        print_table(results)

    feasible = sum(1 for r in results if r["recommendation"]["level1"] is not None)
    deltas = [r["recommendation"]["new_childcare_min"] - r["recommendation"]["original_childcare_min"]
              for r in results]
    print(f"\n{len(results)} recommendation(s): {feasible} feasible, "
          f"median delta {np.median(deltas):.0f} min")

    if args.out:
        payload = {
            "baseline": args.baseline,
            "clustering": args.clustering,
            "k_clusters": int(clusterer.k),
            "n_pool": len(all_ids),
            "sleep_median": sleep_median,
            "results": results,
        }
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
