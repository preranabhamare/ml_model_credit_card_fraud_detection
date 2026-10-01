"""
Real-Time Fraud Detection - Offline Benchmarking & Pre-training Script
Executes full research paper pipeline:
1. Loads 284,807 transactions from Kaggle CSV (or synthetic generator)
2. Drops 'Time' attribute
3. Standardizes 'Amount' using StandardScaler
4. Stratified Train-Test split (80/20)
5. Applies SMOTE to rebalance training data
6. Trains:
   - Logistic Regression (Baseline)
   - Random Forest (max_depth=25)
   - XGBoost (max_depth=6)
   - CatBoost (max_depth=8)
7. Evaluates Research Paper Optimization Targets:
   - Sub-150ms prediction latency
   - High specificity (>99.9%)
   - Minimized Brier Score (<0.0005)
8. Persists models and benchmark metrics to disk for instantaneous dashboard loading.
"""
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DEFAULT_KAGGLE_PATH,
    MODELS_DIR,
    SLA_MAX_LATENCY_MS,
    SPECIFICITY_TARGET,
    BRIER_SCORE_TARGET
)
from data_pipeline import load_dataset, preprocess_pipeline
from model_manager import get_model_instances, benchmark_models, CATBOOST_AVAILABLE

def run_pretraining_and_benchmark(csv_path=DEFAULT_KAGGLE_PATH, force_retrain=False):
    print("=" * 70)
    print("REAL-TIME FRAUD DETECTION: TRAINING & BENCHMARKING ENGINE")
    print("=" * 70)
    t0 = time.time()
    df, is_synthetic = load_dataset(csv_path)
    print(f"Loaded dataset with {len(df):,} transactions. Synthetic: {is_synthetic}. Time: {time.time()-t0:.2f}s")
    
    fraud_count = int(df["Class"].sum())
    legit_count = len(df) - fraud_count
    fraud_rate = (fraud_count / len(df)) * 100
    print(f"Class Distribution: {legit_count:,} Legitimate vs {fraud_count:,} Fraud ({fraud_rate:.3f}%)")
    print("\nExecuting Preprocessing Pipeline (Drop 'Time', Scale 'Amount', SMOTE train)...")
    pipe_out = preprocess_pipeline(df, test_size=0.20, apply_smote_flag=True, smote_sampling_strategy=0.25)
    
    X_train = pipe_out["X_train"]
    y_train = pipe_out["y_train"]
    X_test = pipe_out["X_test"]
    y_test = pipe_out["y_test"]
    scaler = pipe_out["scaler"]

    print(f"Original Train: {pipe_out['original_train_counts']}")
    print(f"SMOTE Resampled Train: {pipe_out['resampled_train_counts']}")
    print(f"Holdout Test Set: {pipe_out['test_counts']}")
    scaler_path = MODELS_DIR / "amount_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Amount StandardScaler saved to: {scaler_path}")
    test_samples_df = X_test.copy()
    test_samples_df["Class"] = y_test
    test_samples_df["Amount_Raw"] = pipe_out["X_test_raw"]["Amount"].values
    fraud_samples = test_samples_df[test_samples_df["Class"] == 1].head(20)
    legit_samples = test_samples_df[test_samples_df["Class"] == 0].head(20)
    
    sample_pool = {
        "fraud": fraud_samples.to_dict(orient="records"),
        "legit": legit_samples.to_dict(orient="records")
    }
    with open(MODELS_DIR / "sample_test_pool.json", "w") as f:
        json.dump(sample_pool, f)
    print(f"Sample injection pool saved to: {MODELS_DIR / 'sample_test_pool.json'}")
    models = get_model_instances()
    trained_models = {}

    print("\nTraining Benchmark Models...")
    for name, model in models.items():
        clean_name = name.lower().replace(" ", "_")
        model_file = MODELS_DIR / f"{clean_name}_model.joblib"

        if model_file.exists() and not force_retrain:
            print(f"  [LOADED] Loading pre-trained {name} from disk...")
            try:
                trained_models[name] = joblib.load(model_file)
                continue
            except Exception as e:
                print(f"      Failed loading {name}, retraining: {e}")

        print(f"  [*] Training {name} on {len(X_train):,} resampled instances...")
        t_m0 = time.time()
        model.fit(X_train, y_train)
        dur = time.time() - t_m0
        print(f"  [OK] {name} trained in {dur:.2f}s")
        joblib.dump(model, model_file)
        trained_models[name] = model
    print("\nBenchmarking Models on Holdout Test Set against Research Paper SLAs...")
    benchmark_data = benchmark_models(trained_models, X_test, y_test)
    print("-" * 110)
    print(f"{'Model':<22} | {'Latency (ms)':<14} | {'Specificity':<12} | {'Brier Score':<12} | {'Accuracy':<10} | {'Recall':<8} | {'Precision':<10} | {'PR-AUC':<8}")
    print("-" * 110)
    for name, res in benchmark_data.items():
        lat_str = f"{res['avg_latency_ms']:.2f}ms"
        spec_str = f"{res['specificity']*100:.3f}%"
        brier_str = f"{res['brier_score']:.6f}"
        acc_str = f"{res['accuracy']*100:.2f}%"
        rec_str = f"{res['recall']*100:.2f}%"
        prec_str = f"{res['precision']*100:.2f}%"
        prauc_str = f"{res['pr_auc']:.4f}"
        print(f"{name:<22} | {lat_str:<14} | {spec_str:<12} | {brier_str:<12} | {acc_str:<10} | {rec_str:<8} | {prec_str:<10} | {prauc_str:<8}")
    print("-" * 110)
    benchmark_file = MODELS_DIR / "benchmark_results.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark results successfully persisted to: {benchmark_file}")
    metadata = {
        "dataset_path": str(csv_path),
        "is_synthetic": is_synthetic,
        "total_records": len(df),
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate_pct": fraud_rate,
        "test_size": len(X_test),
        "test_fraud_count": int(y_test.sum()),
        "sla_max_latency_ms": SLA_MAX_LATENCY_MS,
        "specificity_target": SPECIFICITY_TARGET,
        "brier_score_target": BRIER_SCORE_TARGET,
        "catboost_available": CATBOOST_AVAILABLE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Engine Ready] All models, scalers, and benchmark metrics prepared.")
    return benchmark_data, metadata

if __name__ == "__main__":
    run_pretraining_and_benchmark()
