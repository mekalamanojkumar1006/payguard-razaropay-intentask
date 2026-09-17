"""
PayGuard Orchestrator — Agent Pipeline.

Coordinates the multi-stage risk workflow:

  Transaction → Signal Extraction → ML Scoring → Risk Assessment
  → Decision → Investigation Report

All stages run inside the same process — no separate microservices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

from backend.config import PayGuardConfig, DEFAULT_CONFIG
from backend.features.signal_engine import extract_signals
from backend.services.risk_engine import compute_risk_score, RiskAssessment
from backend.services.decision_engine import make_decision, Decision
from backend.agents.investigator import (
    InvestigationReport,
    get_investigator,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full result from the orchestration pipeline."""
    transaction_id: str
    risk_score: float
    risk_level: str
    ml_probability: float
    decision: str
    risk_signals: list
    investigation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PayGuardOrchestrator:
    """
    Central orchestrator that drives the analysis pipeline.

    Requires a trained sklearn model (Pipeline) that exposes `predict_proba`.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: list,
        cfg: PayGuardConfig | None = None,
        llm_api_key: Optional[str] = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.cfg = cfg or DEFAULT_CONFIG
        self.investigator = get_investigator(api_key=llm_api_key)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze(self, transaction: Dict[str, Any]) -> PipelineResult:
        """Run the full pipeline on a single transaction dict."""
        tx_id = transaction.get("transaction_id", "UNKNOWN")
        logger.info("Pipeline started for transaction %s", tx_id)

        # 1. Extract risk signals
        signals = extract_signals(transaction, self.cfg)

        # 2. ML probability
        ml_prob = self._predict(transaction)

        # 3. Risk scoring
        assessment = compute_risk_score(ml_prob, signals, self.cfg)

        # 4. Decision
        decision = make_decision(assessment, self.cfg)

        # 5. Investigation report
        report = self.investigator.investigate(transaction, assessment, decision)

        result = PipelineResult(
            transaction_id=tx_id,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            ml_probability=assessment.ml_probability,
            decision=decision.decision,
            risk_signals=[s.to_dict() for s in signals],
            investigation=report.to_dict(),
        )
        logger.info("Pipeline complete for %s — decision=%s", tx_id, decision.decision)
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _predict(self, transaction: Dict[str, Any]) -> float:
        """Extract the model's fraud probability for a single transaction."""
        import pandas as pd

        row = {col: transaction.get(col) for col in self.feature_columns}
        df = pd.DataFrame([row])
        try:
            proba = self.model.predict_proba(df)
            # proba shape: (1, n_classes), fraud class is index 1
            return float(proba[0][1])
        except Exception:
            logger.exception("ML prediction failed — defaulting to 0.5")
            return 0.5
