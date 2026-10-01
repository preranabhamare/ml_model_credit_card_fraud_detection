"""
Stratified 10-Fold Cross-Validation Suite for Real-Time Fraud Detection
Validates statistical stability across 10 folds with zero data leakage:
- Folds created using StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
- In each fold:
    - StandardScaler is fitted on fold train
    - SMOTE is applied strictly on fold train
    - Model is evaluated on the pristine fold validation set (0.172% real imbalance)
- Reports Mean ± Std for Specificity (>99.9%), Brier Score (<0.0005), PR-AUC, and Accuracy.
"""
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE

from config import DEFAULT_KAGGLE_PATH, MODELS_DIR, PCA_FEATURES, AMOUNT_FEATURE, FEATURE_NAMES
from data_pipeline import load_dataset
from model_manager import get_model_instances, CATBOOST_AVAILABLE

def run_stratified_10_fold_cv(csv_path=DEFAULT_KAGGLE_PATH, sample_size=None, n_splits=10):
    print("=" * 70)
    print(f"RUNNING STRATIFIED {n_splits}-FOLD CROSS-VALIDATION (ZERO-LEAKAGE PIPELINE)")
    print("=" * 70)

    df, is_synthetic = load_dataset(csv_path, sample_size=sample_size)
    print(f"Dataset Size: {len(df):,} transactions (Synthetic: {is_synthetic})")
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    X = df[FEATURE_NAMES].copy()
    y = df["Class"].astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_model_instances()

    cv_summary = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} across {n_splits} folds...")
        fold_metrics = {
            "specificity": [],
            "brier_score": [],
            "pr_auc": [],
            "roc_auc": [],
            "accuracy": [],
            "recall": [],
            "precision": [],
            "latency_ms": []
        }

        t_m_start = time.time()
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx]
            X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
            scaler = StandardScaler()
            X_tr[AMOUNT_FEATURE] = scaler.fit_transform(X_tr[[AMOUNT_FEATURE]])
            X_val[AMOUNT_FEATURE] = scaler.transform(X_val[[AMOUNT_FEATURE]])
            smote = SMOTE(sampling_strategy=0.25, random_state=42)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
            model.fit(X_tr_smote, y_tr_smote)
            t0 = time.perf_counter()
            probs = model.predict_proba(X_val)[:, 1]
            t_infer = (time.perf_counter() - t0) * 1000.0 / len(X_val)
            preds = (probs >= 0.50).astype(int)
            cm = confusion_matrix(y_val, preds)
            tn, fp, fn, tp = cm.ravel()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            brier = float(brier_score_loss(y_val, probs))
            acc = float(accuracy_score(y_val, preds))
            rec = float(recall_score(y_val, preds, zero_division=0))
            prec = float(precision_score(y_val, preds, zero_division=0))
            try:
                pr_auc = float(average_precision_score(y_val, probs))
            except Exception:
                pr_auc = 0.0
            try:
                roc_auc = float(roc_auc_score(y_val, probs))
            except Exception:
                roc_auc = 0.5

            fold_metrics["specificity"].append(spec)
            fold_metrics["brier_score"].append(brier)
            fold_metrics["pr_auc"].append(pr_auc)
            fold_metrics["roc_auc"].append(roc_auc)
            fold_metrics["accuracy"].append(acc)
            fold_metrics["recall"].append(rec)
            fold_metrics["precision"].append(prec)
            fold_metrics["latency_ms"].append(t_infer)

        duration = time.time() - t_m_start
        print(f"  {model_name} completed {n_splits} folds in {duration:.1f}s")
        cv_summary[model_name] = {
            "specificity_mean": float(np.mean(fold_metrics["specificity"])),
            "specificity_std": float(np.std(fold_metrics["specificity"])),
            "brier_mean": float(np.mean(fold_metrics["brier_score"])),
            "brier_std": float(np.std(fold_metrics["brier_score"])),
            "pr_auc_mean": float(np.mean(fold_metrics["pr_auc"])),
            "pr_auc_std": float(np.std(fold_metrics["pr_auc"])),
            "roc_auc_mean": float(np.mean(fold_metrics["roc_auc"])),
            "roc_auc_std": float(np.std(fold_metrics["roc_auc"])),
            "accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
            "accuracy_std": float(np.std(fold_metrics["accuracy"])),
            "recall_mean": float(np.mean(fold_metrics["recall"])),
            "recall_std": float(np.std(fold_metrics["recall"])),
            "precision_mean": float(np.mean(fold_metrics["precision"])),
            "precision_std": float(np.std(fold_metrics["precision"]))
        }
    print("\n" + "=" * 90)
    print(f"STRATIFIED {n_splits}-FOLD CV RESULTS (MEAN ± STD)")
    print("=" * 90)
    print(f"{'Model Architecture':<22} | {'Specificity':<20} | {'Brier Score':<20} | {'PR-AUC':<18}")
    print("-" * 90)
    for name, s in cv_summary.items():
        spec_str = f"{s['specificity_mean']*100:.3f}% ± {s['specificity_std']*100:.3f}%"
        brier_str = f"{s['brier_mean']:.6f} ± {s['brier_std']:.6f}"
        prauc_str = f"{s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f}"
        print(f"{name:<22} | {spec_str:<20} | {brier_str:<20} | {prauc_str:<18}")
    print("=" * 90)
    out_file = MODELS_DIR / "cv_10_fold_results.json"
    with open(out_file, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"Results saved to: {out_file}")

    return cv_summary

if __name__ == "__main__":
    run_stratified_10_fold_cv(sample_size=60000, n_splits=10)
