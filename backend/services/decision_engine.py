"""
PayGuard Decision Engine.

Converts a RiskAssessment into a deterministic, testable action decision.
No LLM is used — the mapping is purely rule-based and configurable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

from backend.config import PayGuardConfig, DEFAULT_CONFIG
from backend.services.risk_engine import RiskAssessment

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Final decision output for a transaction."""
    decision: str        # APPROVE / VERIFY / BLOCK / MANUAL_REVIEW
    risk_level: str
    risk_score: float
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_decision(
    assessment: RiskAssessment,
    cfg: PayGuardConfig | None = None,
) -> Decision:
    """
    Map a risk assessment to a decision using the configured policy.

    Additional override: if ≥4 signals triggered AND score ≥ 60,
    escalate VERIFY → MANUAL_REVIEW for extra caution.
    """
    cfg = cfg or DEFAULT_CONFIG
    policy = cfg.decision_policy

    # Base decision from risk level
    level_map = {
        "LOW": policy.low,
        "MEDIUM": policy.medium,
        "HIGH": policy.high,
    }
    decision_str = level_map.get(assessment.risk_level, "MANUAL_REVIEW")

    # Override: many signals + borderline score → escalate
    n_triggered = len(assessment.triggered_signals)
    if decision_str == "VERIFY" and n_triggered >= 4 and assessment.risk_score >= 60:
        decision_str = "MANUAL_REVIEW"

    # Build reason codes
    reason_codes: List[str] = []
    for sig_name in assessment.triggered_signals:
        reason_codes.append(f"SIGNAL_{sig_name.upper()}")
    if assessment.ml_probability >= 0.5:
        reason_codes.append("ML_HIGH_PROBABILITY")

    dec = Decision(
        decision=decision_str,
        risk_level=assessment.risk_level,
        risk_score=assessment.risk_score,
        reason_codes=reason_codes,
    )
    logger.info("Decision: %s (risk=%s score=%.2f)", dec.decision, dec.risk_level, dec.risk_score)
    return dec
