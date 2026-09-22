# 🔋 Data-Driven State of Charge Estimation for EV Batteries
### Using ECM-Integrated Machine Learning Models

## 📌 Overview

This project develops a **hybrid physics-informed machine learning framework** for accurately estimating the **State of Charge (SOC)** of lithium-ion batteries in electric vehicles (EVs). SOC — the remaining battery capacity as a percentage of full charge — is critical for:

- Preventing overcharging and deep discharge
- Reducing range anxiety for EV drivers
- Extending battery lifespan
- Enabling safe Battery Management System (BMS) operation

Traditional methods like Coulomb counting and OCV-based estimation suffer from drift errors and impracticality in real-time use. This work bridges that gap by combining an **Equivalent Circuit Model (ECM)** with state-of-the-art ML architectures.

---

## 🧠 Models Implemented

| Model | Type | Key Strength |
|-------|------|--------------|
| **Random Forest (RF)** | Ensemble / Tree-based | Highest accuracy, interpretable |
| **Bidirectional LSTM (BiLSTM)** | Deep Learning / Recurrent | Captures forward & backward temporal patterns |
| **Transformer Encoder** | Deep Learning / Attention | Best generalization, long-range dependencies |

---

## 📊 Results Summary

| Model | Dataset | RMSE (%) | MAE (%) | R² |
|-------|---------|----------|---------|-----|
| Random Forest | Training | 0.1575 | 0.0984 | 0.9998 |
| Random Forest | Testing | 0.2328 | 0.1272 | 0.9994 |
| BiLSTM | Training | 0.4676 | 0.3121 | 0.9972 |
| BiLSTM | Testing | 0.5071 | 0.3782 | 0.9968 |
| Transformer | Training | 0.4030 | 0.2841 | 0.9980 |
| Transformer | Testing | 0.4357 | 0.3204 | 0.9985 |

> ✅ All three models achieve R² > 0.996 on unseen test data.

---

## 🗂️ Project Structure

```
SOC-Estimation/
│
├── data/
│   ├── raw/                        # Raw BMW i3 driving trip CSVs
│   └── processed/                  # Cleaned, normalized datasets
│
├── matlab/
│   ├── ecm_simulation.m            # 2nd-order RC ECM implementation
│   ├── ekf_parameter_estimation.m  # Extended Kalman Filter
│   ├── preprocessing.m             # Noise filtering, feature extraction
│   └── random_forest_model.m       # RF training via fitrensemble
│
├── python/
│   ├── preprocessing.py            # Data cleaning, normalization, windowing
│   ├── feature_engineering.py      # Derivative features, ECM state integration
│   ├── bilstm_model.py             # BiLSTM architecture & training
│   ├── transformer_model.py        # Transformer Encoder architecture & training
│   ├── evaluation.py               # RMSE, MAE, MAPE, R² metrics
│   └── pca_correlation.py          # PCA and correlation analysis
│
├── results/
│   ├── figures/                    # SOC prediction plots, correlation matrix, PCA plots
│   └── metrics/                    # Model performance tables
│
├── report/
│   └── thesis.pdf                  # Full B.Tech thesis report
│
└── README.md
```

---

## 🔬 Methodology

### 1. Dataset
- **Source:** BMW i3 Battery Dataset — [IEEE Dataport](https://ieee-dataport.org/)
- **Coverage:** 32 real-world driving trips under varying conditions
- **Split:** 22 trips → Training | 10 trips → Testing

### 2. Input Features

| Feature | Physical Role |
|---------|--------------|
| Battery Voltage (V) | SOC-linked electrochemical state |
| Battery Current (I) | Charge/discharge rate |
| Battery Temperature | Thermal effect on capacity |
| Ambient Temperature | Environmental thermal load |
| Velocity | Driving load & power draw |
| Elevation | Regenerative braking trigger |
| Throttle | Driver demand proxy |
| Motor Torque | Mechanical power output |
| Acceleration | Energy demand |
| Regenerative Braking | Energy recovery indicator |
| Air Conditioning Power | Auxiliary electrical load |
| Heater Current | Auxiliary electrical load |
| dV/dt | Voltage rate of change |
| dI/dt | Current rate of change |
| V1, V2 (ECM states) | Polarization voltages from ECM |

### 3. Data Preprocessing
- Missing value and outlier removal (NaN/Inf filtering)
- Moving average smoothing for voltage & current signals
- Z-score normalization for BiLSTM and Transformer (tree-based RF skips this)
- Sliding window sequence construction: **30 timesteps (BiLSTM)** / **60 timesteps (Transformer)**

### 4. Physics Module — Equivalent Circuit Model (ECM)
A **2nd-order RC ECM** models battery dynamics:

```
Terminal Voltage:  V_model = OCV - V1 - V2 - I·R0
OCV-SOC Relation:  OCV = 3.7 + 0.1 × SOC
RC Dynamics:       dV1/dt = -V1/(R1·C1) + I/C1
                   dV2/dt = -V2/(R2·C2) + I/C2
Thermal Model:     dT/dt  = (Q_gen - Q_loss) / (m·Cp)
```

### 5. Parameter Estimation — Extended Kalman Filter (EKF)
EKF estimates internal battery states in real time:
- **State vector:** `[SOC, V1, V2, T]`
- **Measurement:** Terminal voltage
- Includes temperature-dependent resistance updates

### 6. Train/Test Split Strategy
| Model | Strategy |
|-------|----------|
| Random Forest | Random 70/30 holdout |
| BiLSTM & Transformer | Block-shuffled split (80/20) — preserves temporal order within blocks, ensures full SOC range in both sets |

---

## 🏗️ Model Architectures

### Random Forest
- 300 bagged decision trees (`fitrensemble`, MATLAB)
- 12 raw features, scale-invariant

### BiLSTM
```
Input (15 features × 30 timesteps)
→ BiLSTM(128) + LayerNorm + Dropout(0.30)
→ BiLSTM(64, last timestep) + Dropout(0.20)
→ FC(64, ReLU) → Dropout(0.15) → FC(32, ReLU) → Output(1)
Optimizer: Adam | LR: 1e-3 | Batch: 512 | Early Stop: patience=10
```

### Transformer Encoder
```
Input (15 features × 60 timesteps)
→ Linear Projection (d_model=128) + Sinusoidal Positional Encoding
→ 4× [Multi-Head Attention (8 heads) + FFN(256, GELU) + Add & Norm]
→ Mean Pooling → FC(64, GELU) → FC(32, GELU) → Output(1)
Optimizer: AdamW | LR: 1e-4 | Warmup: 5ep | Cosine Decay | Batch: 512
```

---

## 📈 Key Findings

- **Random Forest** achieves the lowest raw error but risks overfitting (R² near 1.0)
- **Transformer** shows the best generalization with minimal train-test performance gap
- **Battery Voltage** is the most dominant SOC-influencing feature across all models
- **ECM features** (V1, V2) improve physical interpretability but require careful calibration — naive integration degraded performance
- **Mechanical features** (Torque, Acceleration) have the least impact on SOC prediction
- PCA confirms that 5 principal components capture ~80% of total data variance

---

## ⚙️ Setup & Usage

### Prerequisites
**Python:**
```bash
pip install numpy pandas scikit-learn tensorflow torch matplotlib seaborn
```

**MATLAB:**
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox

### Running the Pipeline

**Step 1 — Preprocess Data**
```bash
python python/preprocessing.py --data_dir data/raw/ --out_dir data/processed/
```

**Step 2 — Run ECM + EKF (MATLAB)**
```matlab
run matlab/ecm_simulation.m
run matlab/ekf_parameter_estimation.m
```

**Step 3 — Train Models**
```bash
# BiLSTM
python python/bilstm_model.py --data data/processed/ --epochs 50

# Transformer
python python/transformer_model.py --data data/processed/ --epochs 60
```

**Step 4 — Evaluate**
```bash
python python/evaluation.py --model transformer --data data/processed/
```

---

## 📉 Performance Metrics Used

| Metric | Formula |
|--------|---------|
| MAE | (1/N) Σ \|yᵢ - ŷᵢ\| |
| RMSE | √[(1/N) Σ (yᵢ - ŷᵢ)²] |
| MAPE | (100/N) Σ \|yᵢ - ŷᵢ\| / yᵢ |
| R² | 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)² |

---

## 📚 References

- Lin, S.-L. (2024). Deep learning-based SOC estimation for EVs. *Heliyon*, 10:e35780
- Xiong et al. (2014). Data-driven multi-scale EKF for Li-ion batteries. *Applied Energy*, 113:463–476
- Plett, G.L. (2004). Extended Kalman filtering for BMS. *Journal of Power Sources*, 134:252–261
- Li & Wang (2020). BiLSTM-based SOC estimation. *Energy Reports*, 6:1801–1810
- Chen & Zhao (2022). Transformer-based battery state estimation. *IEEE Trans. Industrial Electronics*, 69(5)

---