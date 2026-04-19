import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def _save_and_close(filename, show=False):
    """Save the current figure to outputs/ and optionally display it in a notebook."""
    out_dir = os.path.join(os.path.dirname(__file__), '../outputs')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()


def evaluate_model(model, X_train, X_test, y_train, y_test,
                   tick_labels=None, show_plots=False, threshold=0.5):
    """
    Evaluate a binary classifier with stratified cross-validation and holdout metrics.

    Parameters
    ----------
    threshold : float
        Probability cutoff applied to TEST predictions only (default 0.5).
        A value below 0.5 (e.g. 0.4) increases Recall at the cost of Precision.
        Cross-validation always runs at 0.5 for fair model comparison.
    """
    model_name = model.__class__.__name__
    print(f"\n{model_name}  (threshold={threshold})")

    # 1. Stratified cross-validation on the training set
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'}
    cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
    print(f"  CV   | Prec: {cv['test_precision'].mean():.3f}  "
          f"Rec: {cv['test_recall'].mean():.3f}  "
          f"AUC: {cv['test_roc_auc'].mean():.3f}")

    # 2. Fit on full training data
    model.fit(X_train, y_train)

    # 3. Predictions — train at default 0.5, test at clinical threshold
    y_train_pred = model.predict(X_train)
    if threshold != 0.5 and hasattr(model, 'predict_proba'):
        y_test_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    else:
        y_test_pred = model.predict(X_test)

    # 4. Metrics
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec  = recall_score(y_test, y_test_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_test_pred)

    roc_auc_test = None
    if hasattr(model, 'predict_proba'):
        roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"  Test | Prec: {prec:.3f}  Rec: {rec:.3f}  "
          f"Acc: {acc:.3f}  AUC: {roc_auc_test:.3f}")
    print(f"\n  Classification Report (Test, threshold={threshold}):\n"
          f"{classification_report(y_test, y_test_pred, target_names=['Healthy', 'Heart Disease'])}")

    # 5. Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test  = confusion_matrix(y_test, y_test_pred)

    if tick_labels is None:
        tick_labels = ['Healthy', 'Heart Disease']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=tick_labels,
                yticklabels=tick_labels, ax=axs[0])
    axs[0].set_title(f"{model_name} — Train")
    sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=tick_labels,
                yticklabels=tick_labels, cmap='crest', ax=axs[1])
    axs[1].set_title(f"{model_name} — Test (threshold={threshold})")
    for ax in axs:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    fname = f"{model_name}_confusion_matrix_t{threshold}.png"
    _save_and_close(fname, show=show_plots)

    return {
        'accuracy':   acc,
        'precision':  prec,
        'recall':     rec,
        'roc_auc':    roc_auc_test,
        'cv_roc_auc': cv['test_roc_auc'].mean(),
    }


def plot_importance(model, X, show_shap=True, show_plots=False):
    """
    Visualise feature coefficients, importances, and SHAP beeswarm for a fitted model.
    """
    model_name    = model.__class__.__name__
    feature_names = X.columns

    # Case 1: Logistic Regression — coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        sorted_idx   = np.argsort(np.abs(coefficients))[::-1]
        plt.figure(figsize=(12, 8))
        plt.barh(feature_names[sorted_idx], coefficients[sorted_idx], color='steelblue')
        plt.title(f"Feature Coefficients — {model_name}")
        plt.xlabel("Coefficient Value")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        _save_and_close(f"{model_name}_importance_coef.png", show=show_plots)

    # Case 2: Tree-based models — feature importances
    elif hasattr(model, 'feature_importances_'):
        imp_df = (pd.DataFrame({'feature': feature_names,
                                'importance': model.feature_importances_})
                    .sort_values('importance', ascending=False))
        plt.figure(figsize=(12, 8))
        sns.barplot(data=imp_df, x='importance', y='feature',
                    hue='feature', palette='viridis', legend=False)
        plt.title(f"Feature Importances — {model_name}")
        _save_and_close(f"{model_name}_importance_feature.png", show=show_plots)

    # Case 3: SHAP beeswarm
    if show_shap:
        try:
            if 'Forest' in model_name or 'XGB' in model_name:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]
                shap.summary_plot(shap_vals, X, plot_type='dot', show=False)
            else:
                explainer = shap.Explainer(model, X)
                shap_vals = explainer(X)
                shap.plots.beeswarm(shap_vals, max_display=20, show=False)
            _save_and_close(f"{model_name}_shap_beeswarm.png", show=show_plots)
        except Exception as e:
            print(f"SHAP could not be computed for {model_name}: {e}")


def get_mismatches(model, X, y, df_original, threshold=0.5):
    """Return rows from df_original where model prediction differs from the true label."""
    if threshold != 0.5 and hasattr(model, 'predict_proba'):
        y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    mismatch_mask = (y != y_pred)
    mismatches = df_original[mismatch_mask].copy()
    if not mismatches.empty:
        mismatches['predicted_diagnosis'] = y_pred[mismatch_mask]
        return mismatches
    return None
