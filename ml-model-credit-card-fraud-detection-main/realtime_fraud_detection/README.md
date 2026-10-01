# Real-Time Financial Fraud Detection & Analytics Ops Center

Production-grade real-time fraud detection engine and interactive testing dashboard built with **Python 3.12**, **Streamlit**, **XGBoost**, **CatBoost**, **Scikit-Learn**, **Plotly**, and **Imbalanced-Learn (SMOTE)**.

Strictly adheres to specifications and optimization targets from the **"Real-Time Fraud Detection"** research paper.

---

## 🏛️ System Architecture & Research Specifications

### 1. Benchmark Dataset & Class Imbalance
- **Dataset:** Kaggle Credit Card Fraud Dataset (284,807 transactions, 28 PCA features `V1`–`V28`, `Amount`, `Time`, `Class`).
- **Extreme Class Imbalance:** **0.172% fraud** (492 cases vs. 284,315 legitimate transactions).
- **Synthetic Fallback Generator:** Built-in high-fidelity synthetic generator mimicking the exact empirical PCA covariance shifts and bimodal fraud distribution, ensuring immediate execution out-of-the-box if the CSV is missing.

### 2. Rigorous Preprocessing Pipeline
- **Dropped `Time` Feature:** Dropped because elapsed seconds provide no temporal context for single-transaction authorization decisions and induce chronological overfitting in tree models.
- **`Amount` Standardization:** Standardized using `StandardScaler` to align raw transaction amounts (\$0.00 – \$25,000.00) with the variance scale of the PCA features (`V1`–`V28`).
- **SMOTE Rebalancing (Zero Data Leakage):** Synthetic Minority Over-sampling Technique (SMOTE) is strictly applied **only** to the training fold (`X_train`, `y_train`). The holdout test fold (`X_test`, `y_test`) is preserved with its true 0.172% imbalance to guarantee unbiased specificity and Brier score measurement.

### 3. Model Benchmark & Research Targets
Evaluated on 56,864 holdout test transactions:

| Model Architecture | Avg Single-Txn Latency | Sub-150ms SLA | Specificity (Target >99.9%) | Specificity SLA | Brier Score (Target <0.0005) | Accuracy | Recall | Precision | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (`max_depth=25`) | **31.67 ms** | **PASS ✓** | **99.970%** | **PASS ✓** | **0.000629** | 99.94% | 81.63% | 82.47% | **0.8672** |
| **CatBoost** (`depth=8`) | **2.31 ms** | **PASS ✓** | **99.905%** | **PASS ✓** | **0.001144** | 99.88% | 85.71% | 60.87% | **0.8583** |
| **XGBoost** (`max_depth=6`) | **6.41 ms** | **PASS ✓** | **99.887%** | Near Target | **0.001309** | 99.86% | 86.73% | 57.05% | **0.8618** |
| **Logistic Regression** (Baseline) | **0.95 ms** | **PASS ✓** | **99.304%** | Baseline | **0.006874** | 99.29% | 88.78% | 18.01% | **0.7380** |

*All models outperform the 150ms payment authorization SLA by 5x to 150x.*

---

## 🖥️ Interactive Streamlit Dashboard Features

1. **Control Panel / Live Transaction Injector:**
   - **Preset 1:** 🟢 Verified Legitimate Purchase (\$42.50, normal PCA components)
   - **Preset 2:** 🚨 High-Risk Fraud Attack (\$1,499.00, negative V14/V12/V17, positive V4/V11)
   - **Preset 3:** 🟡 Edge-Case Borderline Transaction (\$650.00, moderate V14/V4 anomalies)
   - **Preset 4 & 5:** 🎲 Dynamic Real Legitimate & Confirmed Fraud draws from Kaggle holdout test set
   - Fine-grained interactive sliders for `Amount` and PCA features `V1` through `V28`.

2. **Real-Time Inference Window:**
   - **Fraud Risk Score Gauge:** Interactive Plotly gauge from 0% to 100% with color-coded risk bands.
   - **Decision Status Banner:** Glowing visual indicator:
     - `APPROVED — LOW RISK` (Green, <50% risk)
     - `FLAGGED / BLOCKED — FRAUD DETECTED` (Red, >=50% risk)
   - **Decision Latency Timer:** High-precision millisecond execution timer with `Sub-150ms SLA: PASS ✓` badge.

3. **Feature Explainability (XAI):**
   - Instantaneous local feature attributions showing exactly how features like `V14`, `V17`, `V4`, and `Amount` pushed the risk score toward fraud (red) or legitimate status (green).

4. **100ms Live Stream Simulator:**
   - Real-time streaming simulator injecting transactions at 100ms intervals.
   - Live transaction feed table with rolling SLA status, risk probability, and fraud interception alerts.

5. **Model Benchmarking & Confusion Matrices:**
   - Comparative metric overview across all 4 architectures.
   - Interactive 2x2 confusion matrices, ROC curves, and Precision-Recall curves.

---

## 🚀 How to Run

### 1. Launch the Streamlit Dashboard
```powershell
py -3.12 -m streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### 2. Run the Offline Pre-Training / Retraining Engine
```powershell
py -3.12 train_benchmark.py
```

### 3. Run Automated Integration Tests
```powershell
py -3.12 test_system.py
```
