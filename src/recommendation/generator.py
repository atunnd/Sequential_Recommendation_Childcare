import numpy as np
from src.config import N_CAT

CATEGORY_NAMES = {
    0: "Sleep", 1: "Grooming/Personal Care", 2: "Housework/Food Prep",
    3: "Home Maintenance", 4: "Childcare (HH)", 5: "HH Adult Care",
    6: "Non-HH Care", 7: "Work", 8: "Job Search/Work-Related",
    9: "Education", 10: "Shopping", 11: "Professional/Gov Services",
    12: "Eating/Drinking", 13: "Socializing", 14: "TV/Screen/Leisure",
    15: "Exercise/Sports", 16: "Religious/Volunteer", 17: "Travel", 18: "Other",
}

CHILDCARE_CAT = 4
TV_CAT = 14

CHILDCARE_CODE_NAMES = {
    "030101": "Physical care for HH children",
    "030102": "Reading to/with HH children",
    "030103": "Playing with HH children",
    "030104": "Arts and crafts with HH children",
    "030105": "Playing sports with HH children",
    "030108": "Organization/planning for HH children",
    "030109": "Looking after HH children",
    "030110": "Attending HH children's events",
    "030111": "Waiting for/with HH children",
    "030112": "Picking up/dropping off HH children",
    "030186": "Talking with/listening to HH children",
    "030199": "Childcare for HH children, NEC",
    "030201": "Homework (HH children)",
    "030202": "Meetings and school conferences (HH children)",
    "030203": "Home schooling of HH children",
    "030204": "HH children's homework, NEC",
    "030299": "Helping HH children, NEC",
    "030301": "Providing medical care to HH children",
    "030302": "Obtaining medical care for HH children",
    "030303": "Waiting associated with HH children's health",
    "030399": "HH children's health, NEC",
}


def _transition_matrix(seq: dict, n_cat: int = N_CAT) -> np.ndarray:
    count = np.zeros((n_cat, n_cat))
    acts = seq["activities"]
    for s, d in zip(acts[:-1], acts[1:]):
        if 0 <= s < n_cat and 0 <= d < n_cat:
            count[s, d] += 1
    row_sums = count.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return count / row_sums


def _seq_to_str(activities: list) -> str:
    return " -> ".join(CATEGORY_NAMES.get(a, str(a)) for a in activities)


def generate_recommendation(
    query_id: str,
    exemplar_ids: list,
    sequences: dict,
    max_edits: int = 2,
) -> dict:
    """
    Two-level childcare recommendation with up to max_edits slot substitutions.

    Returns:
        query_id, level1 (best transition), level2 (specific activity code),
        context (exemplar gap), exemplar_count, original_sequence, new_sequence,
        original_childcare_min, new_childcare_min, edit_distance.
    """
    query_seq = sequences[query_id]
    query_acts = query_seq["activities"]
    query_durs = query_seq["durations"]
    original_cc = int(query_seq["childcare_min"])
    original_sequence = _seq_to_str(query_acts)

    if not exemplar_ids:
        return {
            "query_id": query_id, "level1": None, "level2": None,
            "context": {"childcare_gap_minutes": 0}, "exemplar_count": 0,
            "original_sequence": original_sequence,
            "new_sequence": original_sequence,
            "original_childcare_min": original_cc,
            "new_childcare_min": original_cc,
            "edit_distance": 0,
        }

    # Context: gap between query and exemplar pool childcare time
    exemplar_cc = float(np.mean([sequences[uid]["childcare_min"] for uid in exemplar_ids]))
    childcare_gap = int(exemplar_cc - original_cc)

    # Transition matrices
    query_present = set(a for a in query_acts if 0 <= a < N_CAT)
    query_trans = _transition_matrix(query_seq)
    exemplar_trans = np.mean(
        [_transition_matrix(sequences[uid]) for uid in exemplar_ids], axis=0
    )
    trans_gap = exemplar_trans - query_trans

    # Rank all source activities by their gap toward childcare (Fix 2)
    src_gaps = []
    for src in query_present:
        gap = float(trans_gap[src, CHILDCARE_CAT])
        if gap > 0:
            src_gaps.append((src, gap))
    src_gaps.sort(key=lambda x: x[1], reverse=True)
    top_sources = src_gaps[:max_edits]

    # Level 1: best source (for reporting)
    level1 = None
    if top_sources:
        best_src, _ = top_sources[0]
        displaced = int(np.argmax(query_trans[best_src]))
        level1 = {
            "source":      CATEGORY_NAMES.get(best_src, str(best_src)),
            "source_id":   int(best_src),
            "displaced":   CATEGORY_NAMES.get(displaced, str(displaced)),
            "displaced_id": int(displaced),
            "target":      "Childcare (HH)",
            "target_id":   CHILDCARE_CAT,
        }

    # Level 2: most frequent 6-digit childcare code at best src→Childcare transition
    level2 = None
    if level1 is not None:
        best_src = level1["source_id"]
        code_counts: dict = {}
        for uid in exemplar_ids:
            seq = sequences[uid]
            acts = seq["activities"]
            raw = seq.get("raw_codes", [])
            for i in range(len(acts) - 1):
                if acts[i] == best_src and acts[i + 1] == CHILDCARE_CAT and i + 1 < len(raw):
                    code = str(raw[i + 1])
                    code_counts[code] = code_counts.get(code, 0) + 1
        if code_counts:
            total = sum(code_counts.values())
            best_code = max(code_counts, key=code_counts.get)
            level2 = {
                "code": best_code,
                "activity_name": CHILDCARE_CODE_NAMES.get(best_code, "Childcare activity"),
                "frequency": round(code_counts[best_code] / total, 3),
            }

    # Apply up to max_edits substitutions using stored IDs (Fix 2 + Fix 3)
    new_acts = list(query_acts)
    new_durs = list(query_durs)
    new_cc = original_cc
    edited_slots = set()

    for src, _ in top_sources:
        displaced = int(np.argmax(query_trans[src]))
        for i in range(len(new_acts) - 1):
            if (new_acts[i] == src and new_acts[i + 1] == displaced
                    and (i + 1) not in edited_slots):
                new_acts[i + 1] = CHILDCARE_CAT
                new_cc += int(new_durs[i + 1])
                edited_slots.add(i + 1)
                break

    def _activity_minutes(acts, durs):
        totals = {}
        for a, d in zip(acts, durs):
            name = CATEGORY_NAMES.get(a, str(a))
            totals[name] = totals.get(name, 0) + int(d)
        return totals

    return {
        "query_id": query_id,
        "level1": level1,
        "level2": level2,
        "context": {"childcare_gap_minutes": childcare_gap},
        "exemplar_count": len(exemplar_ids),
        "original_sequence": original_sequence,
        "new_sequence": _seq_to_str(new_acts),
        "original_childcare_min": original_cc,
        "new_childcare_min": new_cc,
        "edit_distance": len(edited_slots),
        "original_activity_minutes": _activity_minutes(query_acts, query_durs),
        "new_activity_minutes": _activity_minutes(new_acts, new_durs),
    }
