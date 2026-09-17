"""
PayGuard Feature Engine — Risk Signal Extraction.

Transforms a raw transaction dict into structured, interpretable risk signals.
Every signal carries: triggered (bool), severity (0-1), and a human-readable evidence string.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

from backend.config import PayGuardConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal data structure
# ---------------------------------------------------------------------------

@dataclass
class RiskSignal:
    """One risk signal with its assessment."""
    name: str
    triggered: bool
    severity: float  # 0.0 – 1.0
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def amount_anomaly(tx: Dict[str, Any], cfg: PayGuardConfig) -> RiskSignal:
    """Compare amount against historical average."""
    amount = float(tx.get("amount", 0))
    hist_avg = float(tx.get("historical_average_amount", 0))
    safe_avg = max(hist_avg, 1.0)  # avoid division by zero
    ratio = amount / safe_avg

    triggered = ratio >= cfg.signal_thresholds.amount_ratio_threshold
    severity = _clamp((ratio - 1.0) / 10.0)  # linear scale: ratio 11 → 1.0
    evidence = f"Amount ${amount:.2f} is {ratio:.1f}x historical average ${safe_avg:.2f}"
    return RiskSignal("amount_anomaly", triggered, round(severity, 4), evidence)


def velocity_anomaly(tx: Dict[str, Any], cfg: PayGuardConfig) -> RiskSignal:
    """Evaluate transaction velocity across time windows."""
    t10 = int(tx.get("transactions_last_10_minutes", 0))
    t1h = int(tx.get("transactions_last_1_hour", 0))
    t24h = int(tx.get("transactions_last_24_hours", 0))

    th = cfg.signal_thresholds
    triggered = (t10 >= th.velocity_10m_threshold or
                 t1h >= th.velocity_1h_threshold or
                 t24h >= th.velocity_24h_threshold)

    # severity = worst-case ratio across windows
    ratios = [
        t10 / max(th.velocity_10m_threshold, 1),
        t1h / max(th.velocity_1h_threshold, 1),
        t24h / max(th.velocity_24h_threshold, 1),
    ]
    severity = _clamp(max(ratios) - 0.5)  # shift so threshold=1 → 0.5
    evidence = f"{t10} txns in 10 min, {t1h} in 1 hr, {t24h} in 24 hr"
    return RiskSignal("velocity_anomaly", triggered, round(severity, 4), evidence)


def device_anomaly(tx: Dict[str, Any], _cfg: PayGuardConfig) -> RiskSignal:
    """Flag if the device changed recently."""
    changed = bool(int(tx.get("device_changed_recently", 0)))
    severity = 0.70 if changed else 0.0
    evidence = "Device changed recently" if changed else "Device unchanged"
    return RiskSignal("device_anomaly", changed, severity, evidence)


def location_anomaly(tx: Dict[str, Any], _cfg: PayGuardConfig) -> RiskSignal:
    """Flag if the location changed recently."""
    changed = bool(int(tx.get("location_changed_recently", 0)))
    severity = 0.75 if changed else 0.0
    evidence = "Location changed recently" if changed else "Location unchanged"
    return RiskSignal("location_anomaly", changed, severity, evidence)


def failed_transaction_signal(tx: Dict[str, Any], _cfg: PayGuardConfig) -> RiskSignal:
    """Assess previous failed transactions."""
    fails = int(tx.get("previous_failed_transactions", 0))
    triggered = fails >= 2
    severity = _clamp(fails / 10.0)
    evidence = f"{fails} previous failed transaction(s)"
    return RiskSignal("failed_transactions", triggered, round(severity, 4), evidence)


def chargeback_signal(tx: Dict[str, Any], _cfg: PayGuardConfig) -> RiskSignal:
    """Assess previous chargeback history."""
    cbs = int(tx.get("previous_chargebacks", 0))
    triggered = cbs >= 1
    severity = _clamp(cbs / 5.0)
    evidence = f"{cbs} previous chargeback(s)"
    return RiskSignal("previous_chargebacks", triggered, round(severity, 4), evidence)


def account_age_signal(tx: Dict[str, Any], cfg: PayGuardConfig) -> RiskSignal:
    """Flag very young accounts."""
    age_days = int(tx.get("customer_age_of_account", 0))
    young = age_days <= cfg.signal_thresholds.account_age_young_days
    severity = _clamp(1.0 - (age_days / 365.0)) if young else 0.0
    evidence = f"Account age: {age_days} day(s)"
    return RiskSignal("account_age", young, round(severity, 4), evidence)


def amount_risk_signal(tx: Dict[str, Any], cfg: PayGuardConfig) -> RiskSignal:
    """Flag absolutely high-value transactions."""
    amount = float(tx.get("amount", 0))
    triggered = amount >= cfg.signal_thresholds.high_amount_absolute
    severity = _clamp((amount - 200.0) / 2000.0) if triggered else 0.0
    evidence = f"Transaction amount ${amount:.2f}"
    return RiskSignal("amount_risk", triggered, round(severity, 4), evidence)


# ---------------------------------------------------------------------------
# Combined behavioral risk
# ---------------------------------------------------------------------------

def combined_behavioral_risk(signals: List[RiskSignal]) -> RiskSignal:
    """
    Combine multiple weak signals.
    If ≥3 signals are triggered, produce a combined risk signal
    whose severity is the mean of the triggered severities.
    """
    triggered_signals = [s for s in signals if s.triggered]
    count = len(triggered_signals)
    triggered = count >= 3
    if count > 0:
        mean_sev = sum(s.severity for s in triggered_signals) / count
    else:
        mean_sev = 0.0
    severity = _clamp(mean_sev)
    evidence = f"{count} individual signal(s) triggered"
    return RiskSignal("combined_behavioral", triggered, round(severity, 4), evidence)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ALL_SIGNAL_EXTRACTORS = [
    amount_anomaly,
    velocity_anomaly,
    device_anomaly,
    location_anomaly,
    failed_transaction_signal,
    chargeback_signal,
    account_age_signal,
    amount_risk_signal,
]


def extract_signals(
    tx: Dict[str, Any],
    cfg: PayGuardConfig | None = None,
) -> List[RiskSignal]:
    """
    Run all signal extractors on a transaction and return the full list,
    including the combined behavioral signal.
    """
    cfg = cfg or DEFAULT_CONFIG
    signals: List[RiskSignal] = []
    for extractor in ALL_SIGNAL_EXTRACTORS:
        try:
            sig = extractor(tx, cfg)
            signals.append(sig)
        except Exception:
            logger.exception("Signal extractor %s failed", extractor.__name__)

    # Add composite signal
    signals.append(combined_behavioral_risk(signals))
    return signals
