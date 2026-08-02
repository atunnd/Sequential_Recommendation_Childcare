import random
from src.config import RANDOM_SEED, TRAIN_FRAC, VAL_FRAC


def split_respondents(
    sequences: dict,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (train_ids, val_ids, test_ids).
    Split is by tucaseid so each respondent's full day stays in one partition.
    """
    ids = list(sequences.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def subset(sequences: dict, ids: list[str]) -> dict:
    return {k: sequences[k] for k in ids}
