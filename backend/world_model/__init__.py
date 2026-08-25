"""
World Model Package for Network Attack Forecasting (SIH Problem Statement 26153)

This package contains:
- mitre_mapper: Maps network attack types to standardized MITRE ATT&CK kill-chain stages
- state_aggregator: Bins flow and packet telemetry into time-windowed network state vectors S_t
- world_model_core: PyTorch LSTM + Multi-Head Attention dynamics model learning P(S_t+1 | S_t)
- forecaster: K-step recursive forward simulation engine for infiltration forecasting
- temporal_explainer: Attention-based and SHAP temporal feature attribution
- benchmark: Quantitative comparison against Logistic Regression baseline
"""

from .mitre_mapper import MITREStage, MITREMapper
from .state_aggregator import NetworkStateAggregator
from .world_model_core import WorldModelDynamics, WorldModelTrainer
from .forecaster import KStepForecaster
from .temporal_explainer import TemporalExplainer
from .benchmark import WorldModelBenchmark
from .attack_chain_predictor import AttackChainPredictor, get_predictor

__all__ = [
    'MITREStage',
    'MITREMapper',
    'NetworkStateAggregator',
    'WorldModelDynamics',
    'WorldModelTrainer',
    'KStepForecaster',
    'TemporalExplainer',
    'WorldModelBenchmark',
    'AttackChainPredictor',
    'get_predictor'
]
