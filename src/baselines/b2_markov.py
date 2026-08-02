import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.config import N_CAT, B2_SVD_COMPONENTS, RANDOM_SEED


def compute_b2_raw(activities: list[int], n_cat: int = N_CAT) -> np.ndarray:
    """
    Returns L2-normalized flattened first-order Markov transition matrix (625,).
    Fix applied: L2-normalize the sparse vector before SVD/clustering to prevent
    distance being dominated by zero/non-zero pattern rather than probabilities.
    """
    count = np.zeros((n_cat, n_cat), dtype=np.float64)
    for src, dst in zip(activities[:-1], activities[1:]):
        if 0 <= src < n_cat and 0 <= dst < n_cat:
            count[src, dst] += 1

    row_sums = count.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1             # avoid division by zero
    prob = count / row_sums

    feat = prob.flatten()
    norm = np.linalg.norm(feat)
    return feat / (norm + 1e-8)             # L2 normalize


def compute_b2_batch(sequences: dict, n_cat: int = N_CAT) -> tuple[list[str], np.ndarray]:
    """Compute raw B2 features for all respondents. Returns (ids, (N, 625) matrix)."""
    ids = list(sequences.keys())
    X = np.array([compute_b2_raw(sequences[k]["activities"], n_cat) for k in ids])
    return ids, X


class B2Pipeline:
    """Wraps B2 feature extraction + TruncatedSVD. Fit on train, transform all splits."""

    def __init__(self, n_components: int = B2_SVD_COMPONENTS):
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)

    def fit_transform(self, sequences_train: dict) -> tuple[list[str], np.ndarray]:
        ids, X_raw = compute_b2_batch(sequences_train)
        X_reduced = self.svd.fit_transform(X_raw)
        return ids, X_reduced

    def transform(self, sequences: dict) -> tuple[list[str], np.ndarray]:
        ids, X_raw = compute_b2_batch(sequences)
        X_reduced = self.svd.transform(X_raw)
        return ids, X_reduced
