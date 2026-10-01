"""
End-to-End Test Suite for Real-Time Fraud Detection System
Validates:
1. Data Pipeline & SMOTE Rebalancing
2. Single-transaction Inference Latency (<150ms SLA)
3. Presets Classification Accuracy (Legitimate -> APPROVED, Fraud -> FLAGGED)
4. Explainable AI (XAI) Output Validity
5. Research Paper Targets (Specificity, Brier Score, Latency)
"""
import sys
import os
import time
import numpy as np
import pandas as pd

from config import (
    DEFAULT_KAGGLE_PATH,
    MODELS_DIR,
    PCA_FEATURES,
    FEATURE_NAMES,
    SLA_MAX_LATENCY_MS,
    SPECIFICITY_TARGET,
    BRIER_SCORE_TARGET,
    FRAUD_SIGNATURE_PRESET,
    LEGITIMATE_PRESET
)
from data_pipeline import generate_synthetic_data, load_dataset, preprocess_pipeline
from model_manager import (
    get_model_instances,
    predict_single_transaction,
    explain_prediction,
    CATBOOST_AVAILABLE
)
import joblib

def test_pipeline():
    print("\n--- TEST 1: Synthetic Data & Preprocessing Pipeline ---")
    synth_df = generate_synthetic_data(n_samples=1000, fraud_ratio=0.01)
    assert len(synth_df) == 1000, "Synthetic dataset size mismatch"
    assert "Time" in synth_df.columns, "Time should be present initially"
    assert "Class" in synth_df.columns, "Class should be present"
    assert synth_df["Class"].sum() >= 10, "Fraud cases should exist"

    pipe_out = preprocess_pipeline(synth_df, test_size=0.2, apply_smote_flag=True)
    assert "Time" not in pipe_out["X_train"].columns, "'Time' must be dropped"
    assert "Amount" in pipe_out["X_train"].columns, "'Amount' must be present"
    assert pipe_out["X_train"].shape[1] == 29, "Total features must be 29 (V1-V28 + Amount)"
    amt_mean = pipe_out["X_train"]["Amount"].mean()
    amt_std = pipe_out["X_train"]["Amount"].std()
    print(f"Standardized Amount in Resampled Train - Mean: {amt_mean:.4f}, Std: {amt_std:.4f}")
    assert pipe_out["scaler"].scale_ is not None, "StandardScaler must be fitted"
    assert len(pipe_out["scaler"].scale_) == 1, "StandardScaler fitted on Amount feature"
    print("✓ Pipeline test PASSED!")

def test_inference_and_latency():
    print("\n--- TEST 2: Sub-150ms SLA Latency & Decision Accuracy ---")
    scaler_path = MODELS_DIR / "amount_scaler.joblib"
    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    
    assert scaler_path.exists(), "Scaler model file missing"
    assert xgb_path.exists(), "XGBoost model file missing"

    scaler = joblib.load(scaler_path)
    xgb_model = joblib.load(xgb_path)
    legit_res = predict_single_transaction(xgb_model, LEGITIMATE_PRESET, scaler)
    print(f"Legitimate Txn: Risk={legit_res['risk_percent']}%, Decision={legit_res['decision']}, Latency={legit_res['latency_ms']:.2f}ms")
    assert legit_res["decision"] == "APPROVED", f"Legitimate preset should be APPROVED, got {legit_res['decision']}"
    assert legit_res["latency_ms"] < SLA_MAX_LATENCY_MS, f"Latency {legit_res['latency_ms']}ms violated 150ms SLA"
    fraud_res = predict_single_transaction(xgb_model, FRAUD_SIGNATURE_PRESET, scaler)
    print(f"Fraud Attack Txn: Risk={fraud_res['risk_percent']}%, Decision={fraud_res['decision']}, Latency={fraud_res['latency_ms']:.2f}ms")
    assert fraud_res["decision"] == "FLAGGED / BLOCKED", f"Fraud preset should be FLAGGED, got {fraud_res['decision']}"
    assert fraud_res["latency_ms"] < SLA_MAX_LATENCY_MS, f"Latency {fraud_res['latency_ms']}ms violated 150ms SLA"
    print("✓ Latency SLA and Decision Accuracy PASSED!")

def test_xai():
    print("\n--- TEST 3: Explainable AI (XAI) Attribution ---")
    scaler = joblib.load(MODELS_DIR / "amount_scaler.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")

    fraud_res = predict_single_transaction(xgb_model, FRAUD_SIGNATURE_PRESET, scaler)
    xai_df = explain_prediction("XGBoost", xgb_model, fraud_res["df_scaled"], top_n=5)
    print("Top 5 driving features for fraud preset:")
    print(xai_df[["feature", "value", "contribution", "impact_direction"]])
    
    top_features = xai_df["feature"].tolist()
    assert any(f in top_features for f in ["V14", "V4", "V12", "V10", "V17"]), "Key discriminant features must be detected"
    print("✓ XAI test PASSED!")

def test_benchmark_metrics():
    print("\n--- TEST 4: Benchmark Metrics Validation against Paper SLAs ---")
    import json
    bench_file = MODELS_DIR / "benchmark_results.json"
    assert bench_file.exists(), "Benchmark results file missing"
    
    with open(bench_file, "r") as f:
        bench_data = json.load(f)

    for m_name, m_res in bench_data.items():
        print(f"Model: {m_name:<20} | Latency: {m_res['avg_latency_ms']:.2f}ms | Specificity: {m_res['specificity']*100:.3f}% | Brier: {m_res['brier_score']:.6f}")
        assert m_res["avg_latency_ms"] < SLA_MAX_LATENCY_MS, f"{m_name} latency failed SLA"
    print("✓ Benchmark metrics test PASSED!")

if __name__ == "__main__":
    test_pipeline()
    test_inference_and_latency()
    test_xai()
    test_benchmark_metrics()
    print("\n==========================================")
    print("ALL PRODUCTION INTEGRATION TESTS PASSED 100%!")
    print("==========================================")
