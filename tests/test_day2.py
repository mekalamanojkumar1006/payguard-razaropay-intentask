"""
PayGuard Day 2 Test Suite.

Tests cover:
  - Feature signals (amount, velocity, device, location)
  - Risk scoring (bounds, levels)
  - Decision engine
  - Investigation agent
  - API validation
  - End-to-end scenarios (LOW / MEDIUM / HIGH)
  - Cost model
"""
import sys
import os
import pytest

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import PayGuardConfig, DEFAULT_CONFIG
from backend.features.signal_engine import (
    amount_anomaly,
    velocity_anomaly,
    device_anomaly,
    location_anomaly,
    failed_transaction_signal,
    chargeback_signal,
    account_age_signal,
    extract_signals,
)
from backend.services.risk_engine import compute_risk_score, RiskAssessment
from backend.services.decision_engine import make_decision
from backend.agents.investigator import DeterministicInvestigator
from backend.evaluation.cost_model import calculate_costs


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cfg():
    return PayGuardConfig()


@pytest.fixture
def low_risk_tx():
    return {
        "transaction_id": "TX-LOW-001",
        "customer_id": "C001",
        "timestamp": "2023-06-01T10:00:00",
        "amount": 25.0,
        "currency": "USD",
        "payment_method": "Credit Card",
        "merchant_category": "Groceries",
        "device_id": "D001",
        "ip_address": "10.0.0.1",
        "country": "US",
        "customer_age_of_account": 730,
        "historical_average_amount": 30.0,
        "transactions_last_10_minutes": 0,
        "transactions_last_1_hour": 1,
        "transactions_last_24_hours": 3,
        "device_changed_recently": 0,
        "location_changed_recently": 0,
        "previous_chargebacks": 0,
        "previous_failed_transactions": 0,
        "account_velocity": 0.05,
    }


@pytest.fixture
def medium_risk_tx():
    return {
        "transaction_id": "TX-MED-001",
        "customer_id": "C002",
        "timestamp": "2023-06-01T10:00:00",
        "amount": 200.0,
        "currency": "USD",
        "payment_method": "PayPal",
        "merchant_category": "Digital Goods",
        "device_id": "D002",
        "ip_address": "10.0.0.2",
        "country": "FR",
        "customer_age_of_account": 60,
        "historical_average_amount": 50.0,
        "transactions_last_10_minutes": 4,
        "transactions_last_1_hour": 7,
        "transactions_last_24_hours": 12,
        "device_changed_recently": 1,
        "location_changed_recently": 0,
        "previous_chargebacks": 0,
        "previous_failed_transactions": 2,
        "account_velocity": 0.5,
    }


@pytest.fixture
def high_risk_tx():
    return {
        "transaction_id": "TX-HIGH-001",
        "customer_id": "C003",
        "timestamp": "2023-06-01T10:00:00",
        "amount": 2500.0,
        "currency": "USD",
        "payment_method": "Crypto",
        "merchant_category": "Digital Goods",
        "device_id": "D003",
        "ip_address": "10.0.0.3",
        "country": "JP",
        "customer_age_of_account": 5,
        "historical_average_amount": 20.0,
        "transactions_last_10_minutes": 10,
        "transactions_last_1_hour": 18,
        "transactions_last_24_hours": 30,
        "device_changed_recently": 1,
        "location_changed_recently": 1,
        "previous_chargebacks": 3,
        "previous_failed_transactions": 5,
        "account_velocity": 2.0,
    }


# ===================================================================
# PART 1 — Signal Tests
# ===================================================================

class TestAmountAnomaly:
    def test_triggers_above_threshold(self, cfg):
        tx = {"amount": 500.0, "historical_average_amount": 50.0}
        sig = amount_anomaly(tx, cfg)
        assert sig.triggered is True
        assert sig.severity > 0

    def test_no_trigger_below_threshold(self, cfg):
        tx = {"amount": 50.0, "historical_average_amount": 50.0}
        sig = amount_anomaly(tx, cfg)
        assert sig.triggered is False

    def test_handles_zero_history(self, cfg):
        tx = {"amount": 100.0, "historical_average_amount": 0.0}
        sig = amount_anomaly(tx, cfg)
        assert sig.severity >= 0  # no crash


class TestVelocityAnomaly:
    def test_triggers_high_10m(self, cfg):
        tx = {"transactions_last_10_minutes": 5, "transactions_last_1_hour": 5, "transactions_last_24_hours": 5}
        sig = velocity_anomaly(tx, cfg)
        assert sig.triggered is True

    def test_no_trigger_low(self, cfg):
        tx = {"transactions_last_10_minutes": 1, "transactions_last_1_hour": 2, "transactions_last_24_hours": 5}
        sig = velocity_anomaly(tx, cfg)
        assert sig.triggered is False


class TestDeviceAnomaly:
    def test_triggers_when_changed(self, cfg):
        sig = device_anomaly({"device_changed_recently": 1}, cfg)
        assert sig.triggered is True
        assert sig.severity == 0.70

    def test_no_trigger_unchanged(self, cfg):
        sig = device_anomaly({"device_changed_recently": 0}, cfg)
        assert sig.triggered is False


class TestLocationAnomaly:
    def test_triggers_when_changed(self, cfg):
        sig = location_anomaly({"location_changed_recently": 1}, cfg)
        assert sig.triggered is True

    def test_no_trigger_unchanged(self, cfg):
        sig = location_anomaly({"location_changed_recently": 0}, cfg)
        assert sig.triggered is False


class TestFailedTransactions:
    def test_triggers_multiple_fails(self, cfg):
        sig = failed_transaction_signal({"previous_failed_transactions": 5}, cfg)
        assert sig.triggered is True
        assert sig.severity == 0.5

    def test_no_trigger_zero(self, cfg):
        sig = failed_transaction_signal({"previous_failed_transactions": 0}, cfg)
        assert sig.triggered is False


class TestChargebackSignal:
    def test_triggers_any_chargeback(self, cfg):
        sig = chargeback_signal({"previous_chargebacks": 1}, cfg)
        assert sig.triggered is True

    def test_no_trigger_zero(self, cfg):
        sig = chargeback_signal({"previous_chargebacks": 0}, cfg)
        assert sig.triggered is False


class TestAccountAge:
    def test_triggers_young_account(self, cfg):
        sig = account_age_signal({"customer_age_of_account": 10}, cfg)
        assert sig.triggered is True

    def test_no_trigger_old_account(self, cfg):
        sig = account_age_signal({"customer_age_of_account": 365}, cfg)
        assert sig.triggered is False


# ===================================================================
# PART 2 — Risk Score
# ===================================================================

class TestRiskScore:
    def test_score_bounded_0_100(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        assert 0 <= assessment.risk_score <= 100

    def test_high_ml_prob_raises_score(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        low = compute_risk_score(0.01, signals, cfg)
        high = compute_risk_score(0.99, signals, cfg)
        assert high.risk_score > low.risk_score

    def test_risk_level_low(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        assert assessment.risk_level == "LOW"

    def test_risk_level_high(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        assert assessment.risk_level == "HIGH"

    def test_valid_risk_levels(self, medium_risk_tx, cfg):
        signals = extract_signals(medium_risk_tx, cfg)
        for prob in [0.0, 0.3, 0.5, 0.7, 1.0]:
            assessment = compute_risk_score(prob, signals, cfg)
            assert assessment.risk_level in ("LOW", "MEDIUM", "HIGH")


# ===================================================================
# PART 3 — Decision Engine
# ===================================================================

class TestDecisionEngine:
    def test_low_risk_approves(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert dec.decision == "APPROVE"

    def test_high_risk_blocks_or_reviews(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert dec.decision in ("BLOCK", "MANUAL_REVIEW")

    def test_valid_decision_values(self, medium_risk_tx, cfg):
        signals = extract_signals(medium_risk_tx, cfg)
        assessment = compute_risk_score(0.5, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert dec.decision in ("APPROVE", "VERIFY", "BLOCK", "MANUAL_REVIEW")

    def test_reason_codes_populated(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert len(dec.reason_codes) > 0


# ===================================================================
# PART 4 — Investigation Agent
# ===================================================================

class TestInvestigation:
    def test_report_structure(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)

        agent = DeterministicInvestigator()
        report = agent.investigate(high_risk_tx, assessment, dec)

        assert report.summary
        assert report.risk_level in ("LOW", "MEDIUM", "HIGH")
        assert 0 <= report.risk_score <= 100
        assert isinstance(report.key_reasons, list)
        assert isinstance(report.evidence, list)
        assert report.recommended_action in ("APPROVE", "VERIFY", "BLOCK", "MANUAL_REVIEW")

    def test_low_risk_report_no_false_reasons(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        dec = make_decision(assessment, cfg)

        agent = DeterministicInvestigator()
        report = agent.investigate(low_risk_tx, assessment, dec)

        # Every key reason must reference an actually triggered signal
        for reason in report.key_reasons:
            # Reasons come from triggered signals, should mention severity or probability
            assert "severity" in reason.lower() or "probability" in reason.lower() or "ml model" in reason.lower()

    def test_evidence_from_actual_data(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)

        agent = DeterministicInvestigator()
        report = agent.investigate(high_risk_tx, assessment, dec)

        # Evidence must not be empty for a high-risk case
        assert len(report.evidence) > 0


# ===================================================================
# PART 5 — API Validation (unit-level)
# ===================================================================

class TestAPIValidation:
    def test_transaction_request_rejects_zero_amount(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="T1", amount=0)

    def test_transaction_request_rejects_negative_amount(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="T1", amount=-10)

    def test_transaction_request_rejects_empty_id(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="", amount=50.0)

    def test_transaction_request_valid(self):
        from backend.main import TransactionRequest
        tx = TransactionRequest(transaction_id="T1", amount=50.0)
        assert tx.transaction_id == "T1"
        assert tx.amount == 50.0


# ===================================================================
# PART 6 — Cost Model
# ===================================================================

class TestCostModel:
    def test_zero_errors_zero_cost(self):
        result = calculate_costs(true_positives=10, false_positives=0, false_negatives=0, true_negatives=100)
        assert result["total_false_positive_cost"] == 0
        assert result["total_fraud_loss"] == 0

    def test_fp_cost_positive(self):
        result = calculate_costs(true_positives=10, false_positives=5, false_negatives=0, true_negatives=100)
        assert result["total_false_positive_cost"] > 0

    def test_fn_cost_positive(self):
        result = calculate_costs(true_positives=10, false_positives=0, false_negatives=3, true_negatives=100)
        assert result["total_fraud_loss"] > 0


# ===================================================================
# PART 7 — End-to-end scenario sanity
# ===================================================================

class TestEndToEnd:
    def test_low_risk_scenario(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert dec.decision == "APPROVE"
        assert assessment.risk_level == "LOW"

    def test_high_risk_scenario(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        assert assessment.risk_level == "HIGH"
        assert dec.decision in ("BLOCK", "MANUAL_REVIEW")
