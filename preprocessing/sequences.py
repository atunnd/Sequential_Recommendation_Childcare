import pandas as pd

CHILDCARE_CAT = 4  # category ID for Childcare (HH) in 19-category scheme


def build_sequences(df: pd.DataFrame) -> dict[str, dict]:
    """
    Returns {tucaseid: {
        "activities":    [int, ...],   # category IDs (0-18)
        "durations":     [int, ...],   # minutes per activity
        "raw_codes":     [str, ...],   # original 6-digit trcodep (parallel to activities)
        "start_times":   [str, ...],
        "childcare_min": int,          # total minutes in Childcare (HH) category
        "trchildnum":    int,          # number of own HH children under 18
        "year":          int,
    }}
    sorted by activity sequence number within each respondent.
    """
    sequences = {}
    grouped = df.sort_values("tuactivity_n").groupby("tucaseid")
    for tucaseid, group in grouped:
        acts = group["category_id"].tolist()
        durs = group["tuactdur24"].tolist()
        childcare_min = sum(d for a, d in zip(acts, durs) if a == CHILDCARE_CAT)
        sequences[tucaseid] = {
            "activities":    acts,
            "durations":     durs,
            "raw_codes":     group["trcodep"].tolist(),
            "start_times":   group["tustarttim"].tolist(),
            "childcare_min": childcare_min,
            "trchildnum":    int(group["trchildnum"].iloc[0]) if "trchildnum" in group.columns else 0,
            "year":          int(group["tuyear"].iloc[0]) if "tuyear" in group.columns else -1,
        }
    return sequences
