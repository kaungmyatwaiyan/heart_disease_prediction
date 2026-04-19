import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_cleaning import load_and_rename, clean_data
from imputer import fit_imputers, apply_imputation
from feature_engineering import encode_categorical, scale_features
from model_trainer import evaluate_model
from tuner import tune_xgboost, tune_random_forest

# --- Constants ---
THRESHOLD = 0.4   # Clinical decision: prioritise recall (sensitivity) over precision
N_TRIALS  = 100   # Optuna trials per model per scenario


def run_scenario(df_raw, scenario_name, tune=True):
    """
    Execute a clinical scenario experiment.
    Each scenario receives independent Optuna tuning for a fair comparison.
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    target = 'diagnosis'
    df = clean_data(df_raw)

    # 1. Scenario-specific feature/data strategy
    if "NO_SLOPE" in scenario_name:
        df = df.drop('st_slope', axis=1)

    if "NO_IMPUTE" in scenario_name:
        n_before = len(df)
        df = df.dropna()
        print(f"  dropna: {n_before} -> {len(df)} rows retained")

    # 2. Stratified train/test split
    df_train_raw, df_test_raw = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target]
    )
    print(f"  Split  | Train: {len(df_train_raw)}  Test: {len(df_test_raw)}")

    # 3. Imputation — fit on train only
    if "NO_IMPUTE" not in scenario_name:
        imputers     = fit_imputers(df_train_raw)
        df_train_imp = apply_imputation(df_train_raw, imputers)
        df_test_imp  = apply_imputation(df_test_raw,  imputers)
    else:
        df_train_imp, df_test_imp = df_train_raw, df_test_raw

    # 4. Encode categorical features
    df_train_enc = encode_categorical(df_train_imp)
    df_test_enc  = encode_categorical(df_test_imp)

    X_train = df_train_enc.drop(target, axis=1)
    y_train = df_train_enc[target]
    X_test  = df_test_enc.drop(target, axis=1)
    y_test  = df_test_enc[target]

    # 5. Scale — fit on train only
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 6. Hyperparameter tuning (Optuna)
    if tune:
        xgb_params = tune_xgboost(X_train_scaled, y_train, n_trials=N_TRIALS)
        rf_params  = tune_random_forest(X_train_scaled, y_train, n_trials=N_TRIALS)
    else:
        xgb_params = {'random_state': 42, 'eval_metric': 'logloss'}
        rf_params  = {'random_state': 42}

    # 7. Train and evaluate all three models
    models = {
        "LR":  LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "RF":  RandomForestClassifier(**rf_params),
        "XGB": XGBClassifier(**xgb_params),
    }

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test,
            threshold=THRESHOLD
        )
        results[name] = metrics

    return results


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../inputs/heart_disease_uci.csv')
    df_raw    = load_and_rename(data_path)

    scenarios = [
        "FULL_IMPUTE",        # All features + imputation
        "NO_SLOPE_IMPUTE",    # st_slope dropped + imputation
        "NO_SLOPE_NO_IMPUTE", # st_slope dropped + rows with NaN dropped
    ]

    summary_data = []

    for scen_name in scenarios:
        res = run_scenario(df_raw.copy(), scen_name, tune=True)
        for model_name, metrics in res.items():
            summary_data.append({
                'Scenario':       scen_name,
                'Model':          model_name,
                'CV_ROC_AUC':     metrics['cv_roc_auc'],
                'Test_ROC_AUC':   metrics['roc_auc'],
                'Test_Precision': metrics['precision'],
                'Test_Recall':    metrics['recall'],
                'Test_Accuracy':  metrics['accuracy'],
            })

    print("\n" + "="*80)
    print(f"FINAL COMPARISON  (test threshold={THRESHOLD})")
    print("="*80)
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.sort_values(
        ['Test_Recall', 'Test_ROC_AUC', 'Test_Precision'], ascending=False
    ).to_string(index=False))


if __name__ == "__main__":
    main()
