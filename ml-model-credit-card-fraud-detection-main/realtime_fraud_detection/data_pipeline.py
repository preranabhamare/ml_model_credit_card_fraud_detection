"""
Real-Time Fraud Detection - Data Engineering & Preprocessing Pipeline
Implements rigorous research paper preprocessing:
1. Load Kaggle Credit Card dataset (or high-fidelity synthetic fallback)
2. Drop 'Time' attribute
3. Standardize 'Amount' using StandardScaler
4. Stratified Train-Test split
5. SMOTE Rebalancing strictly on the training partition (preventing data leakage)
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DEFAULT_KAGGLE_PATH,
    TIME_FEATURE,
    AMOUNT_FEATURE,
    TARGET_FEATURE,
    PCA_FEATURES,
    FEATURE_NAMES
)

def generate_synthetic_data(n_samples: int = 50000, fraud_ratio: float = 0.00172, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic benchmark credit card transactions with statistical characteristics
    matching the Kaggle Credit Card PCA distribution (V1-V28, Amount, Class).
    Used as an immediate fallback if Kaggle CSV is unavailable.
    """
    np.random.seed(random_state)
    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud
    legit_pca = np.random.normal(0, 1.0, size=(n_legit, 28))
    legit_amount = np.random.lognormal(mean=3.5, sigma=1.2, size=(n_legit, 1))
    fraud_pca = np.random.normal(0, 1.5, size=(n_fraud, 28))
    fraud_pca[:, 13] -= 6.5
    fraud_pca[:, 11] -= 5.0
    fraud_pca[:, 9] -= 4.2
    fraud_pca[:, 3] += 3.8
    fraud_pca[:, 10] += 3.2
    fraud_pca[:, 16] -= 5.5
    fraud_amount = np.random.choice([np.random.uniform(1, 10), np.random.uniform(500, 2500)], size=(n_fraud, 1))
    legit_data = np.hstack([legit_pca, legit_amount, np.zeros((n_legit, 1))])
    fraud_data = np.hstack([fraud_pca, fraud_amount, np.ones((n_fraud, 1))])
    
    combined = np.vstack([legit_data, fraud_data])
    np.random.shuffle(combined)

    cols = PCA_FEATURES + [AMOUNT_FEATURE, TARGET_FEATURE]
    df = pd.DataFrame(combined, columns=cols)
    df[TARGET_FEATURE] = df[TARGET_FEATURE].astype(int)
    df[TIME_FEATURE] = np.sort(np.random.uniform(0, 172800, size=len(df)))
    return df

def load_dataset(csv_path: Optional[str] = None, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, bool]:
    """
    Loads transaction dataset from disk. Falls back to synthetic generator if path is missing.
    Returns (DataFrame, is_synthetic_flag).
    """
    path_to_try = csv_path or DEFAULT_KAGGLE_PATH
    if os.path.exists(path_to_try):
        try:
            print(f"[DataPipeline] Loading dataset from: {path_to_try}")
            df = pd.read_csv(path_to_try)
            if sample_size and sample_size < len(df):
                df_fraud = df[df[TARGET_FEATURE] == 1]
                df_legit = df[df[TARGET_FEATURE] == 0].sample(n=sample_size - len(df_fraud), random_state=42)
                df = pd.concat([df_fraud, df_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            return df, False
        except Exception as e:
            print(f"[DataPipeline] Error loading {path_to_try}: {e}. Falling back to synthetic.")
    
    print("[DataPipeline] Kaggle CSV not found at default location. Generating synthetic transactions.")
    df_synthetic = generate_synthetic_data(n_samples=sample_size or 50000)
    return df_synthetic, True

def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.20,
    apply_smote_flag: bool = True,
    smote_sampling_strategy: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes research paper preprocessing:
    1. Drop 'Time' attribute
    2. Fit StandardScaler on Amount (train only to prevent data leakage)
    3. Stratified split into train/test
    4. Apply SMOTE to training data if requested
    """
    df_clean = df.copy()
    if TIME_FEATURE in df_clean.columns:
        df_clean = df_clean.drop(columns=[TIME_FEATURE])

    X = df_clean[FEATURE_NAMES].copy()
    y = df_clean[TARGET_FEATURE].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[AMOUNT_FEATURE] = scaler.fit_transform(X_train[[AMOUNT_FEATURE]])
    X_test_scaled[AMOUNT_FEATURE] = scaler.transform(X_test[[AMOUNT_FEATURE]])

    original_train_counts = {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
    test_counts = {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())}
    if apply_smote_flag:
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        resampled_train_counts = {0: int((y_train_resampled == 0).sum()), 1: int((y_train_resampled == 1).sum())}
    else:
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train
        resampled_train_counts = original_train_counts

    return {
        "X_train": X_train_resampled,
        "y_train": y_train_resampled,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "original_train_counts": original_train_counts,
        "resampled_train_counts": resampled_train_counts,
        "test_counts": test_counts
    }
