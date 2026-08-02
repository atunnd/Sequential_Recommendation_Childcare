import numpy as np
from src.config import TOP_K_EXEMPLARS
from src.baselines.b1_duration import compute_b1

SLEEP_CAT = 0
EATING_CAT = 12


def find_exemplars(
    query_id: str,
    query_label: int,
    sequences: dict,
    all_ids: list[str],
    labels: np.ndarray,
    childcare_scores: np.ndarray,
    childcare_percentile: float = 0.75,
    top_k: int = TOP_K_EXEMPLARS,
    similarity_threshold: float = 0.5,
    sleep_threshold: float = 360.0,
) -> list[str]:
    """
    Return top-k respondents from the same cluster who:
      1. Are in the top childcare_percentile by childcare_min within cluster
      2. Pass feasibility: sleep >= sleep_threshold and have at least one Eating activity
      3. Have B1 cosine similarity > similarity_threshold to the query
      4. Are ranked by childcare_min descending
    """
    id_to_idx = {uid: i for i, uid in enumerate(all_ids)}

    # Step 1: same cluster, top childcare quartile
    cluster_mask = labels == query_label
    cluster_indices = np.where(cluster_mask)[0]
    cluster_scores = childcare_scores[cluster_indices]
    threshold = np.percentile(cluster_scores, childcare_percentile * 100)
    high_care_indices = cluster_indices[cluster_scores >= threshold]
    candidate_ids = [all_ids[i] for i in high_care_indices if all_ids[i] != query_id]

    if not candidate_ids:
        return []

    # Step 2: feasibility filter — adequate sleep + at least one eating activity
    feasible_ids = [
        uid for uid in candidate_ids
        if (sum(d for a, d in zip(sequences[uid]["activities"], sequences[uid]["durations"])
                if a == SLEEP_CAT) >= sleep_threshold
            and any(a == EATING_CAT for a in sequences[uid]["activities"]))
    ]
    if not feasible_ids:
        feasible_ids = candidate_ids  # relax if no one passes

    # Step 3: B1 cosine similarity filter
    query_vec = compute_b1(
        sequences[query_id]["activities"], sequences[query_id]["durations"]
    )
    similarities = []
    for uid in feasible_ids:
        cand_vec = compute_b1(sequences[uid]["activities"], sequences[uid]["durations"])
        sim = float(np.dot(query_vec, cand_vec))   # both L2-normalized
        similarities.append((uid, sim))

    similarities.sort(key=lambda x: -x[1])
    similar_ids = [uid for uid, sim in similarities if sim >= similarity_threshold]
    if not similar_ids:
        similar_ids = [uid for uid, _ in similarities[:top_k]]

    # Step 4: rank by childcare_score descending, return top-k
    similar_ids_set = set(similar_ids)
    ranked = sorted(
        similar_ids_set,
        key=lambda uid: childcare_scores[id_to_idx[uid]],
        reverse=True,
    )
    return ranked[:top_k]
