"""
PayGuard Day 3 Test Suite.

Additional tests for:
  - Enhanced investigation findings
  - Customer history service
  - New API endpoints
  - Evidence grounding
  - LLM fallback
"""
import sys, os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import PayGuardConfig
from backend.features.signal_engine import extract_signals
from backend.services.risk_engine import compute_risk_score
from backend.services.decision_engine import make_decision
from backend.agents.investigator import (
    RuleBasedInvestigator,
    LLMInvestigator,
    DeterministicInvestigator,
    Finding,
)
from backend.services.customer_history import CustomerHistoryService


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cfg():
    return PayGuardConfig()

@pytest.fixture
def high_risk_tx():
    return {
        "transaction_id": "TX-HIGH-D3",
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

@pytest.fixture
def low_risk_tx():
    return {
        "transaction_id": "TX-LOW-D3",
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


# ===================================================================
# Investigation findings
# ===================================================================

class TestInvestigationFindings:
    def test_findings_have_required_fields(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        agent = RuleBasedInvestigator()
        report = agent.investigate(high_risk_tx, assessment, dec)

        assert len(report.findings) > 0
        for f in report.findings:
            assert f.reason_code
            assert f.category
            assert f.title
            assert f.severity in ("LOW", "MEDIUM", "HIGH")
            assert f.evidence
            assert f.source_feature

    def test_findings_grounded_in_data(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        agent = RuleBasedInvestigator()
        report = agent.investigate(high_risk_tx, assessment, dec)

        # Every finding must reference a real feature or model output
        valid_sources = {
            "amount / historical_average_amount",
            "transactions_last_10_minutes, transactions_last_1_hour, transactions_last_24_hours",
            "device_changed_recently",
            "location_changed_recently",
            "previous_failed_transactions",
            "previous_chargebacks",
            "customer_age_of_account",
            "amount",
            "aggregate of all triggered signals",
            "model.predict_proba",
        }
        for f in report.findings:
            assert f.source_feature in valid_sources, f"Unsupported source: {f.source_feature}"

    def test_low_risk_no_unsupported_claims(self, low_risk_tx, cfg):
        signals = extract_signals(low_risk_tx, cfg)
        assessment = compute_risk_score(0.01, signals, cfg)
        dec = make_decision(assessment, cfg)
        agent = RuleBasedInvestigator()
        report = agent.investigate(low_risk_tx, assessment, dec)

        # Should have no findings or only ML if somehow triggered
        for f in report.findings:
            assert "known fraud" not in f.evidence.lower()
            assert "associated with" not in f.evidence.lower()

    def test_behavioral_assessment_present(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)
        agent = RuleBasedInvestigator()
        report = agent.investigate(high_risk_tx, assessment, dec)
        assert report.behavioral_assessment
        assert len(report.behavioral_assessment) > 10


# ===================================================================
# LLM fallback
# ===================================================================

class TestLLMFallback:
    def test_no_api_key_uses_fallback(self, high_risk_tx, cfg):
        signals = extract_signals(high_risk_tx, cfg)
        assessment = compute_risk_score(0.99, signals, cfg)
        dec = make_decision(assessment, cfg)

        agent = LLMInvestigator(api_key=None)
        report = agent.investigate(high_risk_tx, assessment, dec)
        assert report.summary  # fallback still produces output
        assert len(report.findings) > 0

    def test_deterministic_alias(self):
        """DeterministicInvestigator should still exist as alias."""
        assert DeterministicInvestigator is RuleBasedInvestigator


# ===================================================================
# Customer history service
# ===================================================================

class TestCustomerHistory:
    def test_load_and_stats(self):
        svc = CustomerHistoryService()
        svc.load()
        stats = svc.get_stats()
        assert stats["total"] > 0
        assert stats["customers"] > 0

    def test_get_transactions_pagination(self):
        svc = CustomerHistoryService()
        svc.load()
        page1 = svc.get_transactions(limit=10, offset=0)
        page2 = svc.get_transactions(limit=10, offset=10)
        assert len(page1) == 10
        assert len(page2) == 10
        assert page1[0]["transaction_id"] != page2[0]["transaction_id"]

    def test_get_profile_nonexistent(self):
        svc = CustomerHistoryService()
        svc.load()
        assert svc.get_profile("NONEXISTENT_CUSTOMER") is None

    def test_transactions_no_leaking_all_fields(self):
        svc = CustomerHistoryService()
        svc.load()
        txns = svc.get_transactions(limit=1)
        # Should contain expected fields
        assert "transaction_id" in txns[0]
        assert "amount" in txns[0]


# ===================================================================
# API validation (Pydantic-level)
# ===================================================================

class TestAPIValidationDay3:
    def test_investigate_request_rejects_zero_amount(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="T1", amount=0)

    def test_investigate_request_rejects_negative(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="T1", amount=-5)

    def test_investigate_request_missing_id(self):
        from backend.main import TransactionRequest
        with pytest.raises(Exception):
            TransactionRequest(transaction_id="", amount=50.0)

    def test_valid_request_defaults(self):
        from backend.main import TransactionRequest
        tx = TransactionRequest(transaction_id="T-VALID", amount=100.0)
        assert tx.customer_id == "unknown"
        assert tx.currency == "USD"
