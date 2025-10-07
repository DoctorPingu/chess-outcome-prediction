# Chess Outcome Prediction (SAN sequences)

Predict the winner of a chess game **(white vs black; draws removed)** from the **first 30 full moves (60 plies)** using a sequence model over SAN moves (+ optional Elo features).

---

## TL;DR results

- **Val / Test AUC:** ~**0.72 / 0.73**
- **Val / Test accuracy (tuned threshold):** ~**0.656 / 0.657**
- **Best decision threshold (val):** ~**0.44** (vs 0.50 default)
- **Artifacts saved to `results/`:** `best_seq_model.keras`, `seq_report.json`, `label_mapping.json` (and `class_weights.json` if used)

> These numbers reproduce within a few ±0.5 pp depending on the sampled subset and seed.

---

## Project structure

```
.
├─ data/                        # input & derived data
├─ notebooks/
│  ├─ 01_down_sampling.ipynb    # (optional) create manageable subset from raw source
│  ├─ 02_pre-processing.ipynb   # clean/tokenize SAN, add features, export CSV/NPZ
│  └─ 03_model_training.ipynb   # sequence model training + evaluation & plots
├─ results/                     # saved model + reports
├─ requirements.txt             # pip environment
├─ environment.yml              # conda environment
├─ README.md
└─ LICENSE
```

---

## Setup

### Option A — conda *(recommended)*

```bash
conda env create -f environment.yml
conda activate chessml
```

### Option B — pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> GPU is optional; CPU training works (slower).

---

## Data

Place your input CSV in `data/` (e.g. `data/chess_games_subset.csv`).  
The preprocessing notebook outputs:

- `data/chess_games_clean.csv` (sequence-only dataset)
- `data/chess_games_clean_meta.json` (settings & counts)
- *(optional)* `data/chess_boards_8x8xC.npz` (board planes at checkpoints)

**Git LFS for large files**

```bash
git lfs install
git lfs track "data/*.csv" "data/*.npz"
git add .gitattributes
git add data/
git commit -m "Add large data artifacts via LFS"
```

---

## How to run

1. *(Optional)* `notebooks/01_down_sampling.ipynb` — create a reduced subset from a large source file.  
2. `notebooks/02_pre-processing.ipynb` — tokenize SAN, keep first **60 plies**, add simple features (captures/checks/Elo), export clean CSV (+meta).  
3. `notebooks/03_model_training.ipynb` — train the sequence model, search best threshold on **val**, evaluate on **test**, save artifacts & plots.

**Quick CLI sketch (after preprocessing):**

```bash
# (for reference only; training runs inside the notebook)
python - <<'PY'
print("Use 03_model_training.ipynb to train and evaluate the model.")
PY
```

---

## Model (high level)

- **Sequences (SAN):** Embedding → Bi-GRU → **GlobalMaxPool** (or attention pooling)  
- **Numeric features (optional):** small MLP (LayerNorm → Dense → Dropout)  
- **Fusion:** concat(sequence, numeric) → Dense → Sigmoid  
- **Training:** Adam/AdamW, early stopping on `val_loss`, `ReduceLROnPlateau`  
- **Metrics:** `accuracy`, `AUC`; post-hoc **threshold search** on val

---

## Evaluation & outputs

- Printed reports for **val** and **test**: accuracy, macro-F1, confusion matrix.  
- Plots saved/displayed: Accuracy/Loss, ROC (AUC), PR curve, probability histograms.  
- Decision threshold tuned on validation (≈ **0.44**).

Artifacts written to `results/`:

```
best_seq_model.keras
seq_report.json        # { val: {...}, test: {...}, threshold }
label_mapping.json     # {"black":0,"white":1}
class_weights.json     # present only if class weights used
```

---

## Reproducibility & notes

- Default seed: **42** (`numpy`, `tensorflow`).  
- With larger or different subsets, expect slight variance in metrics.  
- Draws removed; binary target is **white vs black**.  
- For higher accuracy/AUC: more data, stronger sequence encoders, richer features, or board-state inputs.

---

## Environment maintenance

Re-export your current environment if you update packages:

```bash
# conda
conda env export --no-builds | findstr /V "prefix:" > environment.yml

# pip (inside the env)
pip freeze > requirements.txt
```

---

## License & citation

- License: see `LICENSE` (e.g., MIT).  
- If you use this work, please cite the repository and the original game data source used to build the subset.
