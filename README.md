# Heart Disease Prediction Pipeline

Binary classification pipeline to identify heart disease from multi-site clinical data, built for reproducibility, clinical interpretability, and rigorous model comparison.

---

## Clinical Background

The dataset aggregates patient records from four research sites (Cleveland, Hungary, Switzerland, VA Long Beach). The target variable is a binary diagnosis of heart disease.

**Two features are deliberately excluded:**

| Feature | Reason for exclusion |
| :--- | :--- |
| `ca` (major vessels by fluoroscopy) | Directly visualises coronary artery disease — a sub-diagnosis of the target |
| `thal` (thalassemia stress test) | Diagnostic test that confirms the outcome — constitutes target leakage |

Including either would inflate performance metrics without reflecting real-world predictive utility.

---

## Pipeline Design

The pipeline enforces a strict leakage-free order:

```
Raw Data
  → Clean          (remove duplicates, binarise target, replace invalid 0s)
  → Split          (80/20 stratified train/test split)
  → Impute         (fit on train only: IterativeImputer for numeric, SimpleImputer for categorical)
  → Encode         (restore categorical baselines, one-hot encode)
  → Scale          (fit StandardScaler on train only)
  → Tune           (Optuna maximises cross-validated ROC AUC, 100 trials per model)
  → Evaluate       (StratifiedKFold CV + holdout test, threshold=0.4)
```

**Why threshold = 0.4?**  
Lowering the probability cutoff from 0.5 to 0.4 increases Recall (sensitivity) — the model flags more true positives at the cost of some false alarms. In a screening context, missing a sick patient is more costly than an unnecessary referral.

---

## Scenarios

Three clinical scenarios are compared. Each is independently tuned for a fair comparison.

| Scenario | Features | Missing Values |
| :--- | :--- | :--- |
| `FULL_IMPUTE` | All features including `st_slope` | IterativeImputer |
| `NO_SLOPE_IMPUTE` | `st_slope` dropped (>25% missing) | IterativeImputer |
| `NO_SLOPE_NO_IMPUTE` | `st_slope` dropped | Rows with any NaN discarded |

---

## Models

Three classifiers are evaluated per scenario:

| Model | Notes |
| :--- | :--- |
| Logistic Regression | Baseline; `class_weight='balanced'` |
| Random Forest | Tuned via Optuna (includes `max_features`) |
| XGBoost | Tuned via Optuna (includes `reg_alpha`, `min_child_weight`) |

Results are sorted by **Test Recall → Test ROC AUC → Test Precision**.

---

## Project Structure

```
heart_disease_prediction/
├── heart_disease_pipeline.ipynb   # Main presentation notebook (end-to-end)
├── inputs/
│   └── heart_disease_uci.csv      # Raw dataset
├── outputs/                       # Generated plots (gitignored)
├── src/
│   ├── data_cleaning.py           # Load, rename, clean
│   ├── imputer.py                 # Leakage-free fit/transform imputation
│   ├── feature_engineering.py     # Categorical encoding and scaling
│   ├── model_trainer.py           # CV evaluation, SHAP, confusion matrices
│   ├── tuner.py                   # Optuna hyperparameter optimisation
│   └── main.py                    # Script entry point (runs all scenarios)
├── pyproject.toml
└── .gitignore
```

---

## Setup

```bash
# Install dependencies with Poetry
poetry install

# Activate environment
source .venv/bin/activate   # macOS/Linux

# Launch notebook
jupyter notebook heart_disease_pipeline.ipynb
```

---

## Run as Script

```bash
cd src
python main.py
```

This runs all three scenarios (FULL_IMPUTE, NO_SLOPE_IMPUTE, NO_SLOPE_NO_IMPUTE) with full Optuna tuning and prints the final comparison table.

---

## Key Results

After running the pipeline, results are exported to:
- `outputs/scenario_comparison.png` — 4-panel metric comparison (Recall, ROC AUC, Precision, Accuracy)
- `outputs/precision_recall_curve.png` — Threshold analysis for best model
- `outputs/*_shap_beeswarm.png` — Feature attribution for each model
- `outputs/*_confusion_matrix_t0.4.png` — Confusion matrices at clinical threshold

---

## Dependencies

Managed via Poetry. Core packages:

- `scikit-learn >= 1.8`
- `xgboost >= 3.2`
- `optuna >= 4.0`
- `shap`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
