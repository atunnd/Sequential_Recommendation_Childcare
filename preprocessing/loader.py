import pandas as pd
from src.config import ATUSACT_CSV, load_activity_mapping

# atussum has demographics (TESEX, TEAGE, PEEDUCA) + diary metadata
_ATUSSUM_CSV = ATUSACT_CSV.parent.parent / "atussum" / "atussum_0324.csv"


def load_raw(
    act_path=ATUSACT_CSV,
    sum_path=_ATUSSUM_CSV,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per activity record, joined with
    respondent demographics from atussum. Only keeps records with positive duration.
    """
    # TRCODEP = full 6-digit activity code; TRTIER2P = 4-digit tier-2 code (too coarse)
    act_cols = ["TUCASEID", "TUACTIVITY_N", "TRCODEP", "TRTIER2P", "TUACTDUR24", "TUSTARTTIM"]
    sum_cols = ["TUCASEID", "TESEX", "TEAGE", "PEEDUCA", "TUYEAR", "TUDIARYDAY", "TRCHILDNUM"]

    act = pd.read_csv(act_path, usecols=act_cols, dtype={"TRCODEP": str, "TRTIER2P": str})
    act.columns = act.columns.str.lower()
    act = act[act["tuactdur24"] > 0].copy()
    act["trcodep"] = act["trcodep"].str.zfill(6)

    sumdf = pd.read_csv(sum_path, usecols=sum_cols)
    sumdf.columns = sumdf.columns.str.lower()

    df = act.merge(sumdf, on="tucaseid", how="left")
    df = df[df["trchildnum"] > 0].copy()   # parents only (107,584 respondents)
    print(f"  Parent filter (TRCHILDNUM>0): {df['tucaseid'].nunique():,} respondents")
    return df


def map_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer category_id column using the 19-category mapping on TRCODEP."""
    mapping = load_activity_mapping()
    df = df.copy()
    df["category_id"] = df["trcodep"].map(mapping)
    unknown_mask = df["category_id"].isna()
    if unknown_mask.any():
        df.loc[unknown_mask, "category_id"] = 18  # Other/Unknown fallback
    df["category_id"] = df["category_id"].astype(int)
    return df
