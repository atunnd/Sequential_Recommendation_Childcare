# Sequential Recommendation for Childcare Time

A research pipeline over the **American Time Use Survey (ATUS), 2003–2024**, addressing a
behavioural question: *given a parent's 24-hour activity diary, what concrete change would
increase the time they spend on childcare?*

The output is not a score or a label but a **counterfactual edit to the person's actual day** —
for example, *"after Eating/Drinking, do Childcare instead of Housework — specifically, Playing
with HH children (ATUS code 030103)"* — grounded in what behaviourally similar parents actually
did.

The scientific contribution is a controlled **comparison of four ways to represent a day**
(B1–B4). Clustering, exemplar selection, recommendation generation and evaluation are held
identical across all four, so the representation is the only variable.

---

## Table of Contents

- [Problem Setting](#problem-setting)
- [Method](#method)
- [Representations Under Comparison](#representations-under-comparison)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Repository Layout](#repository-layout)
- [Reproducibility](#reproducibility)

---

## Problem Setting

| | |
|---|---|
| **Source** | ATUS 2003–2024 (`atusact_0324.csv`, `atussum_0324.csv`) |
| **Population** | Parents only (`TRCHILDNUM > 0`) — 107,584 respondents |
| **Representation unit** | One respondent-day as an ordered `(activity, duration)` sequence |
| **Activity scheme** | 19 behavioural categories, mapped from 428 raw ATUS codes |
| **Target behaviour** | Childcare (household children), category 4 |
| **Split** | 70 / 15 / 15 by respondent, seeded — 75,308 / 16,137 / 16,139 |

The split is performed **by respondent**, so a person's full day stays inside a single
partition.

---

## Method

The pipeline is a five-stage sequence. Only stage 2 differs between baselines.

1. **Preprocessing** — parse the raw ATUS activity and summary files, map 6-digit codes to the
   19-category scheme, filter to parents, and assemble one ordered sequence per respondent.
2. **Representation** — encode each day as a fixed-length vector (B1–B4; see below).
3. **Clustering** — fit a Bayesian Gaussian Mixture on the training embeddings. The number of
   clusters *K* is **inferred**, not fixed: components below a weight threshold are pruned, so
   each baseline may settle on a different *K*.
4. **Exemplar selection** — for a query respondent, rank cluster peers by childcare minutes
   normalized *within their own cluster*, subject to a feasibility filter that rejects peers
   sleeping below the population median.
5. **Recommendation** — derive the counterfactual edit from the transition statistics of the
   exemplar pool: a source activity, a displaced activity, and a specific ATUS target code.

Encoders and the clusterer are fit on the **training split only**. Validation is held out for
BGMM log-likelihood; recommendations are generated and evaluated on the **test split**.

---

## Representations Under Comparison

| ID | Representation | Dim | Encoder persisted |
|:---|:---|---:|:---:|
| **B1** | Hellinger-transformed duration proportions per category | 19 | stateless |
| **B2** | First-order Markov transition matrix, L2-normalized, reduced by TruncatedSVD | 64 | no |
| **B3** | Transformer encoder pretrained with masked-activity modelling (MAM) | 64 | `b3_model.pt` |
| **B4** | Hybrid — transformer and Markov branches fused | 128 | `b4_model.pt` |

B1 discards order entirely; B2 captures local transitions but not global structure; B3 captures
sequence context; B4 combines both signals.

> **Note on B2.** Its `TruncatedSVD` is fitted at training time and never serialized — only the
> resulting embeddings are. An unseen diary therefore cannot be projected into B2 space without
> refitting, so out-of-sample inference (`--diary`) is supported for B1, B3 and B4 only.

---

## Results

Evaluated on all 16,139 test respondents, BGMM clustering, seed 42.

| Metric | B1 | B2 | B3 | B4 |
|:---|---:|---:|---:|---:|
| Clusters *K* (inferred) | 15 | 20 | 20 | 20 |
| Silhouette score | −0.0117 | −0.0418 | 0.0443 | **0.0835** |
| BGMM val. log-likelihood | 45.57 | 105.40 | 196.29 | **527.00** |
| Feasibility rate | 95.16% | 97.28% | 99.96% | **99.97%** |
| Realized Δ childcare (median) | 90 min | 90 min | **105 min** | **105 min** |
| Childcare introduced rate | 95.16% | 97.27% | 99.96% | **99.97%** |
| Mean exemplar gap | 411.0 min | 363.6 min | 375.8 min | 390.9 min |
| Gap closure (median) | 0.245 | 0.255 | **0.288** | 0.282 |
| Mean edit distance | 1.84 | 1.76 | 1.93 | 1.95 |

**Reading the table.** *Feasibility rate* is the share of respondents for whom any positive
childcare-increasing edit exists. *Gap closure* is the fraction of the peer gap recovered by the
recommended edit. *Edit distance* is the number of changed slots — lower means a less intrusive
suggestion.

The sequence-aware representations (B3, B4) dominate on cluster quality and feasibility, and
recover a larger share of the peer gap, at the cost of a marginally larger edit. Silhouette
scores are near zero throughout, which is expected for high-dimensional behavioural data and
should not be read as cluster separation in the geometric sense.

---

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate           # Linux / macOS: source .venv/bin/activate

pip install -r requirements.txt
```

### Data and pretrained weights

Neither the raw data nor the trained artifacts are tracked in git. Download and unzip both
archives into the repository root:

| Archive | Target directory | Link |
|:---|:---|:---|
| Data | `data_2003_2024/` | [Download](https://drive.google.com/file/d/1TWpUmUea_pjhHHzvD3rCeIHbYW8QXFFT/view?usp=sharing) |
| Weights | `artifacts/` | [Download](https://drive.google.com/file/d/11d0ELuWTliWqjHRh1RJob1AtUDwHVY27/view?usp=sharing) |

The resulting layout must be:

```
data_2003_2024/
├── atusact/atusact_0324.csv
└── atussum/atussum_0324.csv
```

With the weights in place, inference runs without any training step.

---

## Usage

### Training and evaluation

Runs the full pipeline end to end and writes a results file per baseline.

```bash
python run_pipeline.py --baseline b4              # train, cluster, recommend, evaluate
python run_pipeline.py --baseline b3 --skip-train # reuse cached embeddings and clusterer
python run_pipeline.py --baseline b1 --eval-n 500 # evaluate 500 test respondents
```

| Flag | Default | Description |
|:---|:---|:---|
| `--baseline` | `b4` | Representation to run: `b1`, `b2`, `b3`, `b4` |
| `--clustering` | `bgmm` | Clustering algorithm |
| `--skip-train` | off | Load cached model and embeddings instead of refitting |
| `--eval-n` | `100` | Number of test respondents to evaluate |
| `--device` | `cpu` | `cpu` or `cuda` |

### Inference

`scripts/infer.py` produces recommendations from cached artifacts **without retraining**. The
encoder, clusterer and population embeddings are reused as-is.

```bash
python scripts/infer.py --n 5                      # first 5 test respondents
python scripts/infer.py --n 5 --random             # random sample of the test split
python scripts/infer.py --tucaseid 20030404031185  # one known respondent
python scripts/infer.py --diary my_day.json        # an unseen diary (B1/B3/B4 only)
python scripts/infer.py --baseline b3 --n 20 --out recs_b3.json --quiet
```

On Windows, one batch script per baseline runs 20 test respondents and writes a JSON file:

```bat
scripts\infer_b1.bat    :: --baseline b1 --n 20 --out recs_b1.json
scripts\infer_b2.bat
scripts\infer_b3.bat
scripts\infer_b4.bat
```

Each script resolves the repository root itself, so it can be invoked from any directory, and
forwards extra arguments: `scripts\infer_b3.bat --quiet --random`. It selects
`.venv\Scripts\python.exe` when present; set `PYTHON` to override.

#### Unseen diaries

`--diary` accepts a JSON file with durations in minutes, parallel to activities:

```json
{"activities": [0, 2, 14, 4, 7], "durations": [480, 60, 120, 45, 480]}
```

Raw 6-digit ATUS codes may be supplied instead of category ids, and are mapped via
`configs/activity_mapping.yaml`:

```json
{"raw_codes": ["010101", "020201"], "durations": [480, 60]}
```

#### Caching

The exemplar pool is rebuilt on every run, because within-cluster normalization requires every
respondent's diary rather than the query's alone. This means the raw CSVs must be read. Parsed
sequences are therefore memoized in `artifacts/sequences_cache.pkl`, keyed on the size and mtime
of both CSVs, so only the first run pays the full parse. Pass `--no-cache` to bypass it.

---

## Outputs

All artifacts are written to `artifacts/` (git-ignored).

| Artifact | Contents |
|:---|:---|
| `{baseline}_embeddings.pkl` | Train / val / test embeddings with respondent ids |
| `{baseline}_bgmm_model.pkl` | Fitted BGMM clusterer (`bgmm_model.pkl` for B4) |
| `b3_model.pt`, `b4_model.pt` | Encoder checkpoints |
| `{baseline}_results.json` | Metrics and sample recommendations |
| `sequences_cache.pkl` | Memoized parse of the raw CSVs |
| `clusters/cluster_NN.csv` | Per-cluster membership |
| `clusters_*.png` | UMAP, heatmap and childcare-distribution plots |

The exemplar pool is assembled from the **train and test** embeddings, with validation excluded
— identical to `run_pipeline.py`. A query already present in the dataset therefore reproduces the
recommendation in `{baseline}_results.json` exactly.

---

## Repository Layout

```
├── run_pipeline.py              # end-to-end training and evaluation
├── scripts/
│   ├── infer.py                 # inference from cached artifacts
│   └── infer_b{1..4}.bat        # per-baseline inference runners
├── src/
│   ├── config.py                # paths, hyperparameters, activity mapping loader
│   ├── preprocessing/           # loader, sequence construction, splits
│   ├── baselines/               # B1 duration, B2 Markov, B3 transformer, B4 hybrid
│   ├── clustering/              # Bayesian GMM with inferred K
│   ├── recommendation/          # childcare scoring, exemplars, generator
│   └── evaluation/              # automatic metrics
├── configs/
│   └── activity_mapping.yaml    # 428 ATUS codes → 19 categories
├── data_2003_2024/              # raw ATUS extracts (not tracked)
└── artifacts/                   # models, embeddings, results (not tracked)
```

---

## Reproducibility

- All randomness is seeded via `RANDOM_SEED = 42` in `src/config.py`: the respondent split, SVD,
  and encoder initialization.
- The split is deterministic given the seed and the respondent set; changing the raw data changes
  the partitions.
- Encoders and the clusterer are fit on the training split only. Validation is reserved for BGMM
  log-likelihood, and never enters the exemplar pool.
- Hyperparameters are centralized in `src/config.py` rather than passed on the command line.
