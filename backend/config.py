"""
PayGuard Configuration Module.

Centralizes all configurable thresholds, weights, and cost parameters
so they are not scattered throughout business logic.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SignalThresholds:
    """Thresholds that determine when individual risk signals trigger."""
    amount_ratio_threshold: float = 2.5
    velocity_10m_threshold: int = 3
    velocity_1h_threshold: int = 6
    velocity_24h_threshold: int = 15
    account_age_young_days: int = 30
    high_amount_absolute: float = 500.0


@dataclass
class RiskWeights:
    """
    Weights for combining ML probability and deterministic signals
    into a composite risk score (0-100).

    ML probability is the anchor; signal weights add incremental risk.
    The final score is clamped to [0, 100].
    """
    ml_probability: float = 0.55
    amount_anomaly: float = 0.10
    velocity_anomaly: float = 0.10
    device_anomaly: float = 0.05
    location_anomaly: float = 0.05
    previous_chargebacks: float = 0.05
    failed_transactions: float = 0.05
    account_age: float = 0.03
    combined_behavioral: float = 0.02


@dataclass
class RiskLevelThresholds:
    """Thresholds for mapping risk score → risk level."""
    medium: float = 40.0
    high: float = 70.0


@dataclass
class DecisionPolicy:
    """Maps risk levels to default decisions."""
    low: str = "APPROVE"
    medium: str = "VERIFY"
    high: str = "MANUAL_REVIEW"


@dataclass
class CostModel:
    """
    Business cost parameters for false-positive / fraud-loss analysis.
    Values are placeholders — intended to be calibrated on Day 3/4.
    """
    verification_cost: float = 5.0
    manual_review_cost: float = 25.0
    blocked_legitimate_transaction_cost: float = 50.0
    fraud_loss_cost: float = 200.0


@dataclass
class PayGuardConfig:
    """Top-level configuration for the entire PayGuard system."""
    signal_thresholds: SignalThresholds = field(default_factory=SignalThresholds)
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    risk_levels: RiskLevelThresholds = field(default_factory=RiskLevelThresholds)
    decision_policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    cost_model: CostModel = field(default_factory=CostModel)
    model_artifact_dir: str = "backend/models/artifacts"
    data_dir: str = "backend/data"


# Singleton default config — importable anywhere
DEFAULT_CONFIG = PayGuardConfig()
