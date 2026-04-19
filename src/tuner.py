import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Disable heavy logging from Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgboost(X, y, n_trials=50):
    """Finds best hyperparameters for XGBoost using Optuna."""
    print(f"Optimizing XGBoost ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # study.best_params only contains trial.suggest_* params.
    # Fixed params (random_state, eval_metric) must be merged back in manually.
    fixed_params = {'random_state': 42, 'eval_metric': 'logloss'}
    best_params = {**study.best_params, **fixed_params}
    
    print(f"Best ROC AUC: {study.best_value:.4f}")
    return best_params

def tune_random_forest(X, y, n_trials=30):
    """Finds best hyperparameters for Random Forest using Optuna."""
    print(f"Optimizing Random Forest ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        return cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Merge trial params with fixed params
    fixed_params = {'random_state': 42, 'n_jobs': -1}
    best_params = {**study.best_params, **fixed_params}
    
    print(f"Best ROC AUC: {study.best_value:.4f}")
    return best_params
