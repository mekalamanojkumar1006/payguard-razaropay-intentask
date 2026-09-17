"""
PayGuard Investigation Agent — Day 3 Enhanced.

Produces a structured, evidence-grounded investigation report.
Every finding is traceable to actual transaction features.

Architecture:
  InvestigationAgent (ABC)
  +-- RuleBasedInvestigator   (always works offline)
  +-- LLMInvestigator         (pluggable; graceful fallback)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

from backend.services.risk_engine import RiskAssessment
from backend.services.decision_engine import Decision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """One evidence-grounded investigation finding."""
    reason_code: str
    category: str        # AMOUNT / VELOCITY / DEVICE / LOCATION / HISTORY / ML
    title: str
    severity: str        # LOW / MEDIUM / HIGH
    evidence: str
    source_feature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationReport:
    """Full investigation output — useful for both UI and machine processing."""
    summary: str
    risk_score: float
    risk_level: str
    findings: List[Finding]
    behavioral_assessment: str
    recommended_action: str
    # Legacy fields kept for backward compat with Day 2 tests
    key_reasons: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["findings"] = [f.to_dict() if isinstance(f, Finding) else f for f in self.findings]
        return d


# ---------------------------------------------------------------------------
# Category helpers — map signal names to structured findings
# ---------------------------------------------------------------------------

_SIGNAL_CATEGORY: Dict[str, tuple] = {
    "amount_anomaly":      ("AMOUNT",   "AMOUNT_ANOMALY",       "Unusual transaction amount"),
    "velocity_anomaly":    ("VELOCITY", "VELOCITY_ANOMALY",     "High transaction velocity"),
    "device_anomaly":      ("DEVICE",   "DEVICE_CHANGE",        "Recent device change"),
    "location_anomaly":    ("LOCATION", "LOCATION_CHANGE",      "Recent location change"),
    "failed_transactions": ("HISTORY",  "FAILED_TRANSACTIONS",  "Previous failed transactions"),
    "previous_chargebacks":("HISTORY",  "CHARGEBACK_HISTORY",   "Previous chargeback history"),
    "account_age":         ("HISTORY",  "YOUNG_ACCOUNT",        "Young account"),
    "amount_risk":         ("AMOUNT",   "HIGH_VALUE",           "High-value transaction"),
    "combined_behavioral": ("BEHAVIORAL","COMBINED_SIGNALS",    "Multiple risk signals"),
}

_SOURCE_FEATURE: Dict[str, str] = {
    "amount_anomaly":       "amount / historical_average_amount",
    "velocity_anomaly":     "transactions_last_10_minutes, transactions_last_1_hour, transactions_last_24_hours",
    "device_anomaly":       "device_changed_recently",
    "location_anomaly":     "location_changed_recently",
    "failed_transactions":  "previous_failed_transactions",
    "previous_chargebacks": "previous_chargebacks",
    "account_age":          "customer_age_of_account",
    "amount_risk":          "amount",
    "combined_behavioral":  "aggregate of all triggered signals",
}

def _severity_label(severity: float) -> str:
    if severity >= 0.7:
        return "HIGH"
    elif severity >= 0.4:
        return "MEDIUM"
    return "LOW"


def _build_findings(assessment: RiskAssessment) -> List[Finding]:
    """Convert triggered signals into structured findings."""
    findings: List[Finding] = []
    for sig in assessment.signal_details:
        if not sig.get("triggered"):
            continue
        name = sig["name"]
        cat, reason_code, title = _SIGNAL_CATEGORY.get(
            name, ("OTHER", name.upper(), name.replace("_", " ").title())
        )
        findings.append(Finding(
            reason_code=reason_code,
            category=cat,
            title=title,
            severity=_severity_label(sig["severity"]),
            evidence=sig["evidence"],
            source_feature=_SOURCE_FEATURE.get(name, name),
        ))
    return findings


def _behavioral_assessment(assessment: RiskAssessment, tx: Dict[str, Any]) -> str:
    """Compose a one-paragraph behavioral assessment from real data."""
    parts: List[str] = []
    amount = tx.get("amount", 0)
    hist = tx.get("historical_average_amount", 0)
    safe_hist = max(hist, 1.0)
    ratio = amount / safe_hist

    if ratio >= 2.0:
        parts.append(
            f"The transaction amount (${amount:.2f}) is {ratio:.1f}x the customer's "
            f"historical average (${safe_hist:.2f}), indicating a significant spending deviation."
        )
    t10 = tx.get("transactions_last_10_minutes", 0)
    if t10 >= 3:
        parts.append(f"There were {t10} transactions in the last 10 minutes, suggesting elevated velocity.")
    cbs = tx.get("previous_chargebacks", 0)
    if cbs >= 1:
        parts.append(f"The customer has {cbs} previous chargeback(s) on record.")
    fails = tx.get("previous_failed_transactions", 0)
    if fails >= 2:
        parts.append(f"{fails} previous failed transaction(s) increase the risk profile.")
    if tx.get("device_changed_recently"):
        parts.append("A recent device change was detected.")
    if tx.get("location_changed_recently"):
        parts.append("A recent location change was detected.")
    age = tx.get("customer_age_of_account", 999)
    if age <= 30:
        parts.append(f"The account is only {age} day(s) old.")

    if not parts:
        return "No notable behavioral anomalies detected."
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class InvestigationAgent(ABC):
    """Base class for investigation agents."""

    @abstractmethod
    def investigate(
        self,
        transaction: Dict[str, Any],
        assessment: RiskAssessment,
        decision: Decision,
    ) -> InvestigationReport:
        ...


# ---------------------------------------------------------------------------
# Rule-based (offline) investigator
# ---------------------------------------------------------------------------

class RuleBasedInvestigator(InvestigationAgent):
    """Deterministic investigator — always available, no external deps."""

    def investigate(
        self,
        transaction: Dict[str, Any],
        assessment: RiskAssessment,
        decision: Decision,
    ) -> InvestigationReport:
        tx_id = transaction.get("transaction_id", "UNKNOWN")
        findings = _build_findings(assessment)
        behavioral = _behavioral_assessment(assessment, transaction)

        # Build summary
        if not findings:
            summary = (
                f"Transaction {tx_id} appears low-risk. "
                f"No significant risk signals were triggered."
            )
        else:
            top_reasons = ", ".join(f.title for f in findings[:4])
            summary = (
                f"Transaction {tx_id} received a {assessment.risk_level} risk score "
                f"of {assessment.risk_score:.1f}/100. Key concerns: {top_reasons}."
            )

        # ML probability finding
        if assessment.ml_probability >= 0.5:
            findings.append(Finding(
                reason_code="ML_HIGH_PROBABILITY",
                category="ML",
                title="ML model fraud prediction",
                severity=_severity_label(assessment.ml_probability),
                evidence=f"ML model predicts {assessment.ml_probability:.1%} fraud probability",
                source_feature="model.predict_proba",
            ))

        # Legacy compat
        key_reasons = [f"{f.title} -- severity {f.severity}" for f in findings]
        evidence_list = [f.evidence for f in findings]

        report = InvestigationReport(
            summary=summary,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level,
            findings=findings,
            behavioral_assessment=behavioral,
            recommended_action=decision.decision,
            key_reasons=key_reasons,
            evidence=evidence_list,
        )
        logger.info("Investigation complete for %s: %s", tx_id, decision.decision)
        return report


# Keep old name as alias for backward compat
DeterministicInvestigator = RuleBasedInvestigator


# ---------------------------------------------------------------------------
# LLM investigator (pluggable)
# ---------------------------------------------------------------------------

class LLMInvestigator(InvestigationAgent):
    """
    LLM-powered investigator.  Falls back to RuleBasedInvestigator
    when no API key is configured.

    The LLM receives the *structured* evidence from the deterministic
    engine and generates a more natural narrative.  It never overrides
    the factual data.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._fallback = RuleBasedInvestigator()

    def investigate(
        self,
        transaction: Dict[str, Any],
        assessment: RiskAssessment,
        decision: Decision,
    ) -> InvestigationReport:
        if not self._api_key:
            logger.info("No LLM API key -- using rule-based fallback.")
            return self._fallback.investigate(transaction, assessment, decision)

        # Future: call LLM with structured prompt
        logger.warning("LLM integration not yet implemented -- using fallback.")
        return self._fallback.investigate(transaction, assessment, decision)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_investigator(api_key: Optional[str] = None) -> InvestigationAgent:
    """Return the best available investigator."""
    if api_key:
        return LLMInvestigator(api_key=api_key)
    return RuleBasedInvestigator()
