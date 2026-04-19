import pandas as pd
import numpy as np

def load_and_rename(filepath):
    """Loads the UCI Heart Disease dataset and renames columns for clarity."""
    df = pd.read_csv(filepath)
    df.columns = ['id', 'age', 'sex', 'dataset', 'chest_pain_type', 'resting_bp', 'cholesterol',
                  'fbs', 'rest_ecg', 'max_hr', 'exercise_induced_angina',
                  'stress_st_depression', 'st_slope', 'major_vessels', 'thallium_scan', 'diagnosis']
    return df

def clean_data(df):
    """Initial cleaning: duplicates, binary target, and feature adjustments."""
    df = df.drop_duplicates()
    
    # Convert diagnosis to binary (0 = healthy, 1 = heart disease)
    df['diagnosis'] = (df['diagnosis'] > 0).astype(int)
    
    # 1. Map Binary Columns (Baseline: 0)
    # Convert booleans directly to float (True -> 1.0, False -> 0.0, nan -> nan)
    df['fbs'] = df['fbs'].astype(float)
    df['exercise_induced_angina'] = df['exercise_induced_angina'].astype(float)
    
    # 2. Numeric placeholders (0s in these columns are missing data)
    df['resting_bp'] = df['resting_bp'].replace(0, np.nan)
    df['cholesterol'] = df['cholesterol'].replace(0, np.nan)
    
    # Drop non-predictive or leakage-inducing columns.
    # major_vessels and thallium_scan directly visualise coronary disease (target leakage).
    # id and dataset are identifiers, not clinical predictors.
    # st_slope is retained here and dropped selectively in the pipeline scenario logic.
    df = df.drop(['id', 'dataset', 'major_vessels', 'thallium_scan'], axis=1)
    return df
