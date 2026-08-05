import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_2003_2024"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Raw data files
ATUSACT_CSV = DATA_DIR / "atusact" / "atusact_0324.csv"
ATUSSUM_CSV = DATA_DIR / "atussum" / "atussum_0324.csv"

# Constants
N_CAT = 19          # 19 behavioral categories (childcare-focused scheme, 0-indexed)
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15     # test = 1 - TRAIN_FRAC - VAL_FRAC

# Special token IDs (beyond 0..N_CAT-1)
VOCAB_SIZE = N_CAT  # real activity tokens (0-18)
CLS_ID = N_CAT      # 19
MASK_ID = N_CAT + 1 # 20
PAD_ID = N_CAT + 2  # 21
TOTAL_VOCAB = N_CAT + 3  # 22

# Transformer architecture (tiny, CPU-trainable)
TRANS_D_MODEL = 64
TRANS_N_HEADS = 2
TRANS_N_LAYERS = 2
TRANS_FF_DIM = 128
TRANS_DROPOUT = 0.1

# Markov branch
MARKOV_HIDDEN = 128

# Hybrid fused embedding dim
FUSED_DIM = 128

# MAM training
MAM_MASK_RATE = 0.15
MAM_EPOCHS = 10
MAM_BATCH_SIZE = 256
MAM_LR = 1e-3

# Clustering
BGMM_MAX_COMPONENTS = 20
BGMM_WEIGHT_THRESHOLD = 0.01   # components below this weight are pruned
KMEANS_N_INIT = 10

# Recommendation
TOP_K_EXEMPLARS = 20

# SVD for B2
B2_SVD_COMPONENTS = 64

# Ollama (local LLM)
OLLAMA_HOST = "http://localhost:11434"   # default Ollama server address
OLLAMA_AGENT_MODEL = "gemma4"           # model for the recommendation agent
OLLAMA_JUDGE_MODEL = "gemma4"           # model for the LLM judge


def load_activity_mapping() -> dict[str, int]:
    """Return {6-digit-code: category_id} for all ATUS codes."""
    with open(CONFIGS_DIR / "activity_mapping.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mapping = {}
    for cat_id, cat_info in cfg["categories"].items():
        for code in cat_info["codes"]:
            mapping[str(code)] = int(cat_id)
    return mapping
