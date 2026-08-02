import numpy as np
from src.config import N_CAT


def compute_b1(activities: list[int], durations: list[int], n_cat: int = N_CAT) -> np.ndarray:
    """
    Returns a normalized (25,) Hellinger-transformed duration proportion vector.
    Fix applied: sqrt transform + L2 normalize before KMeans to avoid Sleep/Work
    dominance in Euclidean distance (standard practice for compositional data).
    """
    vec = np.zeros(n_cat, dtype=np.float64)
    for act, dur in zip(activities, durations):
        if 0 <= act < n_cat:
            vec[act] += dur

    total = vec.sum()
    if total == 0:
        return vec
    vec /= total                              # proportions on simplex
    vec_h = np.sqrt(vec)                      # Hellinger transform
    norm = np.linalg.norm(vec_h)
    if norm > 0:
        vec_h /= norm                         # L2 normalize
    return vec_h


def compute_b1_batch(sequences: dict, n_cat: int = N_CAT) -> tuple[list[str], np.ndarray]:
    """Compute B1 features for all respondents. Returns (ids, feature_matrix)."""
    ids = list(sequences.keys())
    X = np.array([
        compute_b1(sequences[k]["activities"], sequences[k]["durations"], n_cat)
        for k in ids
    ])
    return ids, X
