# AI-Based Network Attack Forecasting using Causal World Models
> **Smart India Hackathon (SIH) — Problem Statement #26153**  
> *Anticipating Multi-Stage Cyber Kill Chains via Temporal Latent State Transition Dynamics*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏛️ Project Architecture & Directory Structure

```
NIDS-ML/
├── backend/                             # Python Backend & Causal World Model Engine
│   ├── __init__.py
│   ├── api.py                           # REST API Server serving real-time model inference
│   ├── config.py                        # System paths & hyperparameter configurations
│   └── world_model/                     # Core PyTorch AI Engine
│       ├── __init__.py
│       ├── world_model_core.py          # 2-Layer LSTM + Multi-Head Temporal Attention
│       ├── state_aggregator.py          # 30-Dimensional State Vector S_t Ingestion
│       ├── forecaster.py                # K-Step Forward Simulation & Infiltration Probabilities
│       ├── mitre_mapper.py              # MITRE ATT&CK Kill-Chain Stage Alignment
│       ├── attack_chain_predictor.py    # 39-Campaign Markov Transition Model & NCISS Risk Engine
│       ├── temporal_explainer.py        # Gradient Saliency & Attention Attribution (XAI)
│       └── benchmark.py                 # Evaluator vs Logistic Regression & Random Forest
│
├── frontend/                            # Neo-Brutalist Cyber Forensics Web Console
│   ├── index.html                       # Classified Intelligence Dossier UI Layout
│   ├── css/
│   │   └── style.css                    # Archival Cream & Technical Grid Design System
│   └── js/
│       └── app.js                       # Real-Time REST API Client & Interactive Charts
│
├── data/                                # Network Telemetry & Campaign Datasets
│   ├── raw/                             # Multi-stage attack slices from CIC-IDS-2017 (Parquet)
│   └── mitre/                           # 39 MITRE Attack Flow JSONs + NCISS Severity CSV
│
├── models/                              # Serialized Model Artifacts
│   ├── world_model.pt                   # Trained PyTorch Model Weights
│   ├── world_model_config.json          # Architecture Configuration
│   └── world_model_scaler.pkl           # Feature StandardScaler
│
├── results/                             # Benchmark Metrics & Visualizations
│   ├── world_model_benchmark.csv        # Performance comparison table
│   └── plots/                           # ROC curves & confusion matrices
│
├── run_server.py                        # Single-command API backend & frontend launcher
├── train.py                             # Single-command end-to-end model training pipeline
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Technical documentation
```

---

## 🚀 Quickstart: Running the System

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch Defense Console & API Server
```powershell
python run_server.py
```
This boots the REST API backend on port `8000` and automatically opens the interactive dashboard at `http://localhost:8000/`.

### 3. (Optional) Retrain the World Model
```powershell
python train.py
```

---

## 📊 Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score | Lead-Time Capability | Paradigm |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression** (Baseline) | 99.73% | 100.00% | 99.73% | 0.9986 | ❌ 0.0s (Reactive) | Static Flow Classification |
| **Random Forest** (Static ML) | 99.97% | 100.00% | 99.97% | 0.9999 | ❌ 0.0s (Reactive) | Static Flow Classification |
| **Causal World Model** (Ours) | **99.95%** | **100.00%** | **99.95%** | **0.9998** | ✅ **+10.0s – 20.0s** | Causal World Model $P(S_{t+1} \mid S_t)$ |

---

## 🛡️ Key Innovations for SIH Problem Statement #26153

1. **Causal State Transitions $P(S_{t+1} \mid S_t)$**: Rather than classifying isolated flows reactively after an exploit succeeds, our World Model learns temporal transition dynamics over continuous network states.
2. **$K$-Step Forward Simulation**: Generates multi-step threat forecasts up to $K=10$ windows into the future, providing a **10–20 second advance warning**.
3. **Adversary Campaign Priors**: Integrates transition matrices extracted from **39 real-world incident playbooks** (*SolarWinds, Conti, NotPetya, Black Basta*) and NCISS severity scores (0–100).
4. **Interpretable XAI**: Combines temporal attention weights and gradient saliency to explain *why* an escalation is forecast.
5. **100% Open-Source & Offline**: Operates fully on edge/on-premise environments with zero cloud API dependencies.
