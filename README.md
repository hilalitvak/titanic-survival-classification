# Titanic Survival — Classification Pipeline

End-to-end classification pipeline predicting Titanic passenger survival using a PyTorch MLP, with an interactive Streamlit inference UI.

🔗 **Live demo:** [hila-litvak-titanic-survival-classification-model.streamlit.app](https://hila-litvak-titanic-survival-classification-model.streamlit.app/)

---

## Project Structure

```
├── data/
│   └── train.csv              # Downloaded automatically on first run
├── models/
│   └── checkpoint.pkl         # Saved after training
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── src/
│   ├── preprocess.py          # Feature engineering + imputation + scaling
│   ├── model.py               # MLP architecture
│   ├── train.py               # Standalone training script
│   └── evaluate.py            # Metrics + plots (shared by train + app)
├── app/
│   ├── ds_app.py              # Streamlit inference UI
│   └── titanic-sinking.gif    # Header GIF displayed in the app
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

### Kaggle API (recommended)

The training script fetches `train.csv` directly from Kaggle. To enable this:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. Accept the [Titanic competition rules](https://www.kaggle.com/competitions/titanic) (required for API access)

If no Kaggle credentials are found, the data is downloaded from a public GitHub mirror automatically.

---

## Usage

### 1. Train the model

```bash
python src/train.py
```

This will:
- Fetch `data/train.csv` (Kaggle API or GitHub mirror)
- Split into 80% train / 20% validation (stratified, seed=42)
- Preprocess: extract Title from Name, impute Age per-Title median, encode, scale
- Train a 3-layer MLP for 60 epochs
- Print loss + validation accuracy every 10 epochs, then final metrics
- Save `models/checkpoint.pkl`

Example output:
```
✓ Downloaded train.csv from Kaggle API.
Epoch  10/60  loss=0.4821  val_acc=0.8156
Epoch  20/60  loss=0.4103  val_acc=0.8268
...
=== Validation Metrics ===
Accuracy : 0.8324
F1       : 0.7921
ROC-AUC  : 0.8871

Checkpoint saved → models/checkpoint.pkl
```

### 2. Run the Streamlit app

```bash
streamlit run app/ds_app.py
```

The app walks through three explicit steps:

| Step | What happens |
|------|-------------|
| **1 — Provide a dataset** | Upload any Titanic-format CSV, or leave blank to use `data/train.csv` automatically |
| **2 — Load trained model** | Checkpoint is loaded from `models/checkpoint.pkl` and status is confirmed on screen |
| **3 — Run inference** | Click **Run Inference** to generate predictions and view results |

**Results include:**
- 5 metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- 3 plots: Confusion Matrix, ROC Curve, Precision–Recall Curve
- Predictions table: passenger details, ground truth, model prediction, and survival probability

---

## Architecture & Design Choices

### Feature Engineering

| Feature | Source | Rationale |
|---|---|---|
| `Title` | Extracted from Name | Strongest single feature (Cramér's V=0.57); encodes sex + age + social status |
| `Pclass` | Raw | Strong linear signal (r=−0.34) |
| `Age` | Imputed per-Title median | Per-group imputation avoids distorting Master (boys) with the adult median |
| `fare_per_person` | Fare / FamilySize | Normalises group-ticket inflation |
| `FamilySize` | SibSp + Parch + 1 | Non-linear survival curve — mid-size families survived most |
| `has_cabin` | Binary flag from Cabin | 77% missing; missingness itself is informative (1st class had cabins) |
| `Embarked` | One-hot (C/Q/S) | Moderate signal (V=0.17); reflects class mix per port |

**Dropped:** Sex (absorbed by Title), raw Fare, SibSp/Parch, Deck, PassengerId, Name, Ticket.

### Model

3-layer MLP implemented in PyTorch:

```
Input (13) → Linear(64) → ReLU → Dropout(0.3)
           → Linear(32) → ReLU → Dropout(0.3)
           → Linear(1)  → (raw logit)
```

- **Loss:** `BCEWithLogitsLoss` with `pos_weight` (ratio of negatives to positives) to handle class imbalance  
- **Optimiser:** Adam (lr=0.001)  
- **Scheduler:** `ReduceLROnPlateau` — halves LR when val ROC-AUC stops improving (patience=5)  
- **Early stopping:** monitors val ROC-AUC; restores best checkpoint if no improvement for 15 epochs  
- **Max epochs:** 200 (typically converges in 60–120 with early stopping)

### Preprocessing

All imputation values and the StandardScaler are fit on the training split only and saved inside `checkpoint.pkl` alongside the model weights — ensuring identical transformations at inference with no data leakage.

---

## Evaluation

Primary metric: **ROC-AUC** (threshold-independent, robust to class imbalance).  
Secondary: **F1** (balances precision and recall on the slightly imbalanced target).
