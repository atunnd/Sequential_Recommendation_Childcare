"""
Main pipeline: preprocessing → feature extraction → clustering → recommendations → evaluation.

Usage:
    python run_pipeline.py [--baseline b1|b2|b3|b4] [--clustering bgmm] [--skip-train] [--eval-n 50] [--device cpu|cuda]

Clustering modes:
    bgmm          (default) Each baseline independently fits its own BGMM and infers its own K.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.preprocessing.loader import load_raw, map_activities
from src.preprocessing.sequences import build_sequences
from src.preprocessing.splits import split_respondents, subset
from src.baselines.b1_duration import compute_b1_batch
from src.baselines.b2_markov import B2Pipeline
from src.baselines.b3_transformer import B3Pipeline
from src.baselines.b4_hybrid import B4Pipeline
from src.clustering.bgmm import BGMMClustering
from src.recommendation.health_score import compute_childcare_scores
from src.recommendation.exemplars import find_exemplars
from src.recommendation.generator import generate_recommendation
from src.evaluation.auto_eval import evaluate_batch, clustering_silhouette, bgmm_val_loglik

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], default="b4")
    p.add_argument(
        "--clustering", choices=["bgmm"], default="bgmm",
        help="Clustering algorithm: bgmm",
    )
    p.add_argument("--skip-train", action="store_true", help="Load cached model/embeddings")
    p.add_argument("--eval-n", type=int, default=100, help="# test respondents to evaluate")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_or_build_embeddings(baseline: str, sequences_train, sequences_val, sequences_test,
                             skip_train: bool, device: str):
    cache = ARTIFACTS_DIR / f"{baseline}_embeddings.pkl"

    if skip_train and cache.exists():
        print(f"Loading cached {baseline} embeddings from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"\n=== Building {baseline.upper()} features ===")
    if baseline == "b1":
        train_ids, X_train = compute_b1_batch(sequences_train)
        val_ids, X_val = compute_b1_batch(sequences_val)
        test_ids, X_test = compute_b1_batch(sequences_test)

    elif baseline == "b2":
        pipe = B2Pipeline()
        train_ids, X_train = pipe.fit_transform(sequences_train)
        val_ids, X_val = pipe.transform(sequences_val)
        test_ids, X_test = pipe.transform(sequences_test)

    elif baseline == "b3":
        pipe = B3Pipeline()
        train_ids, X_train = pipe.fit_transform(sequences_train, device=device)
        val_ids, X_val = pipe.transform(sequences_val, device=device)
        test_ids, X_test = pipe.transform(sequences_test, device=device)
        pipe.save(str(ARTIFACTS_DIR / "b3_model.pt"))

    elif baseline == "b4":
        pipe = B4Pipeline()
        train_ids, X_train = pipe.fit_transform(sequences_train, device=device)
        val_ids, X_val = pipe.transform(sequences_val, device=device)
        test_ids, X_test = pipe.transform(sequences_test, device=device)
        pipe.save(str(ARTIFACTS_DIR / "b4_model.pt"))

    result = {
        "train": (train_ids, X_train),
        "val": (val_ids, X_val),
        "test": (test_ids, X_test),
    }
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


def main():
    args = parse_args()

    # Step 1-3: Load and preprocess
    print("Loading raw ATUS data...")
    df = load_raw()
    df = map_activities(df)
    print(f"  {len(df)} activity records, {df['tucaseid'].nunique()} respondents")

    sequences = build_sequences(df)

    # Step 4: Split
    train_ids, val_ids, test_ids = split_respondents(sequences)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    seq_train = subset(sequences, train_ids)
    seq_val = subset(sequences, val_ids)
    seq_test = subset(sequences, test_ids)

    # Step 5-9: Feature extraction
    embeddings = load_or_build_embeddings(
        args.baseline, seq_train, seq_val, seq_test, args.skip_train, args.device
    )
    train_ids_emb, X_train = embeddings["train"]
    val_ids_emb, X_val = embeddings["val"]
    test_ids_emb, X_test = embeddings["test"]

    # Clustering 
    bgmm_file = "bgmm_model.pkl" if args.baseline == "b4" else f"{args.baseline}_bgmm_model.pkl"
    bgmm_cache = ARTIFACTS_DIR / bgmm_file
    print(f"\n=== Clustering (BGMM, independent per baseline) for {args.baseline.upper()} ===")

    if args.skip_train and bgmm_cache.exists():
        print("  Loading cached BGMM model...")
        with open(bgmm_cache, "rb") as f:
            clusterer = pickle.load(f)
        k = clusterer.k
    else:
        clusterer = BGMMClustering()
        clusterer.fit(X_train)
        k = clusterer.k
        with open(bgmm_cache, "wb") as f:
            pickle.dump(clusterer, f)
        if args.baseline == "b4":
            with open(ARTIFACTS_DIR / "bgmm_k.json", "w") as f:
                json.dump({"k": k}, f)

    train_labels = clusterer.predict(X_train)
    test_labels = clusterer.predict(X_test)

    all_ids = train_ids_emb + test_ids_emb
    all_embeddings = np.vstack([X_train, X_test])
    all_labels = np.concatenate([train_labels, test_labels])

    # Intrinsic clustering metrics
    print("\n=== Clustering quality metrics ===")
    sil = clustering_silhouette(X_train, train_labels)
    loglik = bgmm_val_loglik(clusterer, X_val) if args.clustering == "bgmm" else None
    print(f"  K (inferred)      : {k}")
    print(f"  Silhouette score  : {sil:.4f}")
    if loglik is not None:
        print(f"  Val log-likelihood: {loglik:.4f}")

    # Childcare scoring 
    print("\n=== Computing childcare scores ===")
    all_childcare_scores = compute_childcare_scores(sequences, all_ids, all_labels)

    # Population sleep median for exemplar feasibility filter
    SLEEP_CAT = 0
    all_sleep = [
        sum(d for a, d in zip(sequences[uid]["activities"], sequences[uid]["durations"])
            if a == SLEEP_CAT)
        for uid in all_ids
    ]
    sleep_median = float(np.median(all_sleep))
    print(f"  Population sleep median: {sleep_median:.0f} min")

    # Recommendations on test set
    eval_n = min(args.eval_n, len(test_ids_emb))
    eval_test_ids = test_ids_emb[:eval_n]
    id_to_all_idx = {uid: i for i, uid in enumerate(all_ids)}

    print(f"\n=== Generating recommendations for {eval_n} test respondents ===")
    recommendations = []
    for uid in eval_test_ids:
        idx = id_to_all_idx[uid]
        cluster_id = int(all_labels[idx])
        exemplar_ids = find_exemplars(
            query_id=uid,
            query_label=cluster_id,
            sequences=sequences,
            all_ids=all_ids,
            labels=all_labels,
            childcare_scores=all_childcare_scores,
            sleep_threshold=sleep_median,
        )
        if not exemplar_ids:
            exemplar_ids = [all_ids[i] for i in
                            np.where(all_labels == cluster_id)[0][:5]
                            if all_ids[i] != uid]
        rec = generate_recommendation(uid, exemplar_ids, sequences)
        recommendations.append(rec)

    # Auto evaluation
    print("\n=== Auto evaluation ===")
    metrics = evaluate_batch(
        test_ids=eval_test_ids,
        sequences=sequences,
        recommendations=recommendations,
    )
    print(f"  Feasibility rate          : {metrics['feasibility_rate']:.2%}")
    print(f"  Realized delta (median)   : {metrics['realized_delta_median']:.1f} min")
    print(f"  Childcare introduced rate : {metrics['childcare_introduced_rate']:.2%}")
    print(f"  Mean exemplar gap         : {metrics['mean_exemplar_gap']:.1f} min")
    print(f"  Gap closure (median)      : {metrics['gap_closure_median']:.3f}")
    print(f"  Mean edit distance        : {metrics['mean_edit_distance']:.2f}")
    print(f"  N evaluated               : {metrics['n_evaluated']}")

    # Save results
    suffix = f"_{args.clustering}" if args.clustering != "bgmm" else ""
    results_path = ARTIFACTS_DIR / f"{args.baseline}{suffix}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "baseline": args.baseline,
            "clustering": args.clustering,
            "k_clusters": int(k),
            "clustering_metrics": {
                "k_clusters": int(k),
                "silhouette_score": round(sil, 4),
                "bgmm_val_loglik": round(loglik, 4) if loglik is not None else None,
            },
            "metrics": metrics,
            "sample_recommendations": recommendations,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
