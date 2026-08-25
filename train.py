"""
NIDS-ML — High-Performance End-to-End Model Training Pipeline
SIH Problem Statement #26153

Trains the Causal World Model (LSTM + Multi-Head Temporal Attention)
on balanced multi-stage attack slices from CIC-IDS-2017, validates state transition
dynamics P(S_t+1 | S_t), and computes the formal benchmark report against Logistic Regression & Random Forest.
"""

import os
import sys
from pathlib import Path
import json
import logging
import time
import numpy as np
import pandas as pd
import torch
import joblib

# Ensure root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.world_model import (
    NetworkStateAggregator,
    WorldModelDynamics,
    WorldModelTrainer,
    KStepForecaster,
    TemporalExplainer,
    WorldModelBenchmark,
    MITREMapper
)

# Setup directories
log_dir = ROOT_DIR / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
models_dir = ROOT_DIR / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
results_dir = ROOT_DIR / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'world_model_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TrainPipeline")


def load_dataset(max_samples_per_file: int = 50000) -> pd.DataFrame:
    """
    Loads balanced multi-stage attack slices from data/raw/
    preserving all rare attack sequences while maintaining reasonable memory.
    """
    raw_dir = ROOT_DIR / 'data' / 'raw'
    files = sorted(list(raw_dir.glob('*.parquet')))
    if not files:
        files = sorted(list(raw_dir.glob('*.csv')))

    if not files:
        raise FileNotFoundError(f"No telemetry files found in {raw_dir}")

    dfs = []
    for f in files:
        logger.info(f"Loading telemetry slice: {f.name}")
        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, low_memory=False)

        # Standardize column whitespace
        df.columns = df.columns.str.strip()

        # Find label column
        label_col = 'Label' if 'Label' in df.columns else ('label' if 'label' in df.columns else None)
        
        if label_col and len(df) > max_samples_per_file:
            # Keep all malicious rows + sample benign to maintain multi-stage representation
            malicious = df[df[label_col].astype(str).str.strip().str.upper() != 'BENIGN']
            benign = df[df[label_col].astype(str).str.strip().str.upper() == 'BENIGN']
            
            n_benign = max(5000, max_samples_per_file - len(malicious))
            sampled_benign = benign.sample(n=min(len(benign), n_benign), random_state=42)
            
            sampled = pd.concat([malicious, sampled_benign]).sort_index()
            dfs.append(sampled)
            logger.info(f"  -> Retained {len(sampled):,} records ({len(malicious):,} malicious, {len(sampled_benign):,} benign)")
        else:
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total curated multi-stage telemetry dataset: {len(combined):,} records")
    return combined


def main():
    logger.info("=" * 65)
    logger.info("Starting Causal World Model Training Pipeline (SIH PS 26153)")
    logger.info("=" * 65)

    # 1. Ingest curated telemetry
    df = load_dataset(max_samples_per_file=50000)

    # 2. State Aggregation & Multi-Scale Feature Engineering
    aggregator = NetworkStateAggregator(window_size=20, sequence_length=10)
    X_seq, y_next_state, y_stages = aggregator.fit_transform(df)
    logger.info(f"Extracted {len(X_seq):,} autoregressive sequences (10 timesteps x 30 features)")

    # 3. Train/Validation/Test Split (70/15/15 chronological)
    n = len(X_seq)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train, y_train_next, y_train_st = X_seq[:train_end], y_next_state[:train_end], y_stages[:train_end]
    X_val, y_val_next, y_val_st       = X_seq[train_end:val_end], y_next_state[train_end:val_end], y_stages[train_end:val_end]
    X_test, y_test_next, y_test_st    = X_seq[val_end:], y_next_state[val_end:], y_stages[val_end:]

    # 4. Model Instantiation & Training
    model = WorldModelDynamics(
        input_dim=30,
        hidden_dim=128,
        num_mitre_stages=6,
        num_lstm_layers=2,
        dropout=0.2
    )

    trainer = WorldModelTrainer(
        model=model,
        lr=3e-3
    )

    history = trainer.fit(
        X_train=X_train,
        y_next_train=y_train_next,
        y_stage_train=y_train_st,
        X_val=X_val,
        y_next_val=y_val_next,
        y_stage_val=y_val_st,
        epochs=8,
        batch_size=128
    )

    # 5. Save Artifacts
    torch.save(model.state_dict(), models_dir / 'world_model.pt')
    with open(models_dir / 'world_model_config.json', 'w') as f:
        json.dump({'input_dim': 30, 'hidden_dim': 128, 'num_mitre_stages': 6}, f, indent=2)
    aggregator.save_scaler(models_dir / 'world_model_scaler.pkl')
    logger.info("Saved trained model weights, config, and scaler to models/")

    # 6. Benchmark Evaluation against Baselines
    logger.info("Executing benchmark comparison against Logistic Regression & Random Forest...")
    X_flat_train = X_train[:, -1, :]
    X_flat_test  = X_test[:, -1, :]

    benchmark = WorldModelBenchmark(world_model=model)
    results_df = benchmark.run_benchmark(
        X_seq_test=X_test,
        y_stage_test=y_test_st,
        X_flat_test=X_flat_test,
        X_flat_train=X_flat_train,
        y_stage_train=y_train_st
    )

    # Save to CSV
    results_df.to_csv(results_dir / 'world_model_benchmark.csv', index=False)
    logger.info(f"Saved benchmark results to {results_dir / 'world_model_benchmark.csv'}")

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION REPORT (SIH PS #26153):")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)
    logger.info("Training & benchmark pipeline completed successfully.")


if __name__ == "__main__":
    main()
