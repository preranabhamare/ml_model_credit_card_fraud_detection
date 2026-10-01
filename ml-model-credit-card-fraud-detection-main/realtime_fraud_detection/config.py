"""
Real-Time Fraud Detection Engine - System Configuration
Specs derived from "Real-Time Fraud Detection" Research Paper & Production SLA Targets.
"""
import os
from pathlib import Path
DEFAULT_KAGGLE_PATH = r"C:\Users\yhing\Desktop\archive (4)\creditcard.csv"
WORKSPACE_DIR = Path(__file__).resolve().parent
MODELS_DIR = WORKSPACE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TIME_FEATURE = "Time"
AMOUNT_FEATURE = "Amount"
TARGET_FEATURE = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_NAMES = PCA_FEATURES + [AMOUNT_FEATURE]
SLA_MAX_LATENCY_MS = 150.0       
SPECIFICITY_TARGET = 0.999       
BRIER_SCORE_TARGET = 0.0005      
DEFAULT_DECISION_THRESHOLD = 0.50
FRAUD_SIGNATURE_PRESET = {
    "V1": -3.04, "V2": 3.16, "V3": -5.17, "V4": 4.41, "V5": -2.31,
    "V6": -1.40, "V7": -4.33, "V8": 1.73, "V9": -2.78, "V10": -5.61,
    "V11": 4.02, "V12": -7.12, "V13": -0.21, "V14": -8.91, "V15": -0.45,
    "V16": -4.61, "V17": -8.24, "V18": -2.89, "V19": 0.81, "V20": 0.35,
    "V21": 0.67, "V22": -0.24, "V23": -0.32, "V24": -0.05, "V25": 0.36,
    "V26": 0.14, "V27": 0.59, "V28": 0.22, "Amount": 1499.00
}

LEGITIMATE_PRESET = {
    "V1": 0.12, "V2": 0.05, "V3": 0.85, "V4": 0.15, "V5": -0.22,
    "V6": 0.08, "V7": 0.14, "V8": 0.02, "V9": 0.11, "V10": -0.05,
    "V11": -0.10, "V12": 0.18, "V13": 0.05, "V14": 0.25, "V15": 0.12,
    "V16": 0.09, "V17": 0.15, "V18": -0.04, "V19": -0.02, "V20": -0.03,
    "V21": -0.02, "V22": 0.01, "V23": 0.03, "V24": 0.02, "V25": -0.01,
    "V26": 0.02, "V27": 0.01, "V28": 0.01, "Amount": 42.50
}

BORDERLINE_PRESET = {
    "V1": -0.95, "V2": 0.85, "V3": -1.20, "V4": 1.75, "V5": -0.45,
    "V6": 0.20, "V7": -1.10, "V8": 0.40, "V9": -0.80, "V10": -1.85,
    "V11": 1.45, "V12": -2.10, "V13": 0.10, "V14": -3.20, "V15": 0.05,
    "V16": -1.50, "V17": -2.40, "V18": -0.90, "V19": 0.30, "V20": 0.15,
    "V21": 0.25, "V22": -0.10, "V23": -0.08, "V24": 0.01, "V25": 0.12,
    "V26": -0.05, "V27": 0.18, "V28": 0.08, "Amount": 650.00
}
COLORS = {
    "bg_dark": "#0B0F19",
    "card_bg": "#111827",
    "card_border": "#1F2937",
    "accent_emerald": "#10B981",    
    "accent_crimson": "#EF4444",    
    "accent_amber": "#F59E0B",      
    "accent_cyan": "#06B6D4",       
    "accent_purple": "#8B5CF6",     
    "text_primary": "#F9FAFB",
    "text_secondary": "#9CA3AF"
}
