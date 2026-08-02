import numpy as np


def normalize_within_cluster(
    scores: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Min-max normalize scores within each cluster independently."""
    normed = np.zeros_like(scores, dtype=np.float64)
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_scores = scores[mask]
        lo, hi = cluster_scores.min(), cluster_scores.max()
        if hi > lo:
            normed[mask] = (cluster_scores - lo) / (hi - lo)
        else:
            normed[mask] = 0.5
    return normed


def compute_childcare_scores(
    sequences: dict, ids: list[str], labels: np.ndarray
) -> np.ndarray:
    """Childcare engagement score = childcare_min, normalized within cluster."""
    raw = np.array([sequences[uid]["childcare_min"] for uid in ids], dtype=np.float64)
    return normalize_within_cluster(raw, labels)
