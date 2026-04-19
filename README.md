# Heart Disease Prediction

Binary classification pipeline using clinical data from four research sites.

## Setup

```bash
poetry install
jupyter notebook heart_disease_pipeline.ipynb
```

## Structure

```
src/
├── data_cleaning.py       # Load and clean raw data
├── imputer.py             # Leakage-free imputation (fit on train only)
├── feature_engineering.py # Encoding and scaling
├── model_trainer.py       # Evaluation, SHAP, confusion matrices
├── tuner.py               # Optuna hyperparameter tuning
└── main.py                # Script entry point
```

## Notes

- `ca` and `thal` excluded — direct diagnostic tests, not predictors.
- Pipeline order: Split → Impute → Encode → Scale.
- Three scenarios compared: Full Impute, No Slope + Impute, No Slope + No Impute.
- Classification threshold set to 0.4 to prioritise recall in a medical context.
