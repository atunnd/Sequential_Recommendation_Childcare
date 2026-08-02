import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.config import BGMM_MAX_COMPONENTS, BGMM_WEIGHT_THRESHOLD, RANDOM_SEED


class BGMMClustering:
    def __init__(
        self,
        max_components: int = BGMM_MAX_COMPONENTS,
        weight_threshold: float = BGMM_WEIGHT_THRESHOLD,
    ):
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            random_state=RANDOM_SEED,
            max_iter=200,
        )
        self.weight_threshold = weight_threshold
        self.k_inferred: int | None = None

    def fit(self, X_train: np.ndarray) -> "BGMMClustering":
        self.bgmm.fit(X_train)

        self.k_inferred = int((self.bgmm.weights_ > self.weight_threshold).sum())
        print(f"BGMM: inferred K = {self.k_inferred} active components")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.bgmm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.bgmm.predict_proba(X)

    @property
    def k(self) -> int:
        assert self.k_inferred is not None, "Call fit() first"
        return self.k_inferred
