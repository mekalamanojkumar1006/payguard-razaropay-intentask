"""
PayGuard Risk Engine.

Combines ML model probability with deterministic risk signals
to produce a normalized risk score (0-100) and a risk level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

from backend.config import PayGuardConfig, DEFAULT_CONFIG
from backend.features.signal_engine import RiskSignal, extract_signals

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a transaction."""
    risk_score: float          # 0 – 100
    ml_probability: float      # 0 – 1
    risk_level: str            # LOW / MEDIUM / HIGH
    triggered_signals: List[str]
    signal_details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_risk_score(
    ml_probability: float,
    signals: List[RiskSignal],
    cfg: PayGuardConfig | None = None,
) -> RiskAssessment:
    """
    Produce a composite risk score from the ML probability and
    deterministic risk signals.

    Strategy:
      score = Σ(weight_i × component_i) × 100
    where each component is in [0, 1].
    ML probability is already in [0, 1].
    Each signal's contribution is its severity if triggered, else 0.
    """
    cfg = cfg or DEFAULT_CONFIG
    w = cfg.risk_weights

    # Map signal names → configured weights
    signal_weight_map: Dict[str, float] = {
        "amount_anomaly": w.amount_anomaly,
        "velocity_anomaly": w.velocity_anomaly,
        "device_anomaly": w.device_anomaly,
        "location_anomaly": w.location_anomaly,
        "previous_chargebacks": w.previous_chargebacks,
        "failed_transactions": w.failed_transactions,
        "account_age": w.account_age,
        "amount_risk": 0.0,  # absorbed into amount_anomaly
        "combined_behavioral": w.combined_behavioral,
    }

    # ML component
    score = w.ml_probability * ml_probability

    # Signal components
    for sig in signals:
        weight = signal_weight_map.get(sig.name, 0.0)
        if sig.triggered:
            score += weight * sig.severity

    # Normalize to 0-100
    risk_score = max(0.0, min(100.0, score * 100.0))

    # Risk level
    thresholds = cfg.risk_levels
    if risk_score >= thresholds.high:
        risk_level = "HIGH"
    elif risk_score >= thresholds.medium:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    triggered = [s.name for s in signals if s.triggered]
    details = [s.to_dict() for s in signals]

    assessment = RiskAssessment(
        risk_score=round(risk_score, 2),
        ml_probability=round(ml_probability, 6),
        risk_level=risk_level,
        triggered_signals=triggered,
        signal_details=details,
    )
    logger.info(
        "Risk assessment: score=%.2f level=%s triggered=%s",
        assessment.risk_score, assessment.risk_level, triggered,
    )
    return assessment
