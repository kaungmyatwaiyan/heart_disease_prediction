import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

def fit_imputers(df_train):
    """Fits numeric and categorical imputers on the training set only."""
    # Identify column types
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Fit Numeric Imputer
    imputer_numeric = IterativeImputer(max_iter=10, random_state=42)
    imputer_numeric.fit(df_train[numeric_cols])

    # Fit Categorical Imputer
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    imputer_categorical.fit(df_train[categorical_cols])
    
    return {
        'numeric': imputer_numeric,
        'categorical': imputer_categorical,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

def apply_imputation(df, imputers):
    """Applies pre-fitted imputers to a dataframe."""
    df_imputed = df.copy()
    
    # Numeric
    num_cols = imputers['numeric_cols']
    df_imputed[num_cols] = imputers['numeric'].transform(df[num_cols])
    
    # Categorical
    cat_cols = imputers['categorical_cols']
    df_imputed[cat_cols] = imputers['categorical'].transform(df[cat_cols])
    
    return df_imputed
