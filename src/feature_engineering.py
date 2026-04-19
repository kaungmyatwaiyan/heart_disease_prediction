import pandas as pd
from sklearn.preprocessing import StandardScaler

def encode_categorical(df):
    """Restores categorical baselines and performs one-hot encoding."""
    df_processed = df.copy()
    
    # 1. Re-apply Categorical Baselines (lost during imputation)
    if 'sex' in df_processed.columns:
        df_processed['sex'] = pd.Categorical(df_processed['sex'], categories=['Female', 'Male'])
    if 'chest_pain_type' in df_processed.columns:
        df_processed['chest_pain_type'] = pd.Categorical(df_processed['chest_pain_type'], 
                                               categories=['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    if 'rest_ecg' in df_processed.columns:
        df_processed['rest_ecg'] = pd.Categorical(df_processed['rest_ecg'], 
                                        categories=['normal', 'st-t abnormality', 'lv hypertrophy'])
    if 'st_slope' in df_processed.columns:
        df_processed['st_slope'] = pd.Categorical(df_processed['st_slope'], 
                                        categories=['upsloping', 'flat', 'downsloping'])

    # 2. One-Hot Encoding
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=int)
    
    # 3. Type conversion for remaining columns
    for col in ['fbs', 'exercise_induced_angina', 'diagnosis']:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(int)
            
    return df_encoded

def scale_features(X_train, X_test):
    """Fits scaler on training data and transforms both train and test sets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
