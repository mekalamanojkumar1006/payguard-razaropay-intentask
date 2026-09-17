"""
PayGuard Cost Model.

Functions for calculating false-positive costs and fraud losses
from evaluation results. Uses configurable CostModel parameters.
"""
from __future__ import annotations

from typing import Dict, Any

from backend.config import CostModel, DEFAULT_CONFIG


def calculate_costs(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int,
    avg_fraud_amount: float = 200.0,
    cost_model: CostModel | None = None,
) -> Dict[str, Any]:
    """
    Calculate business costs from a confusion matrix.

    Args:
        true_positives:  correctly flagged fraud
        false_positives: legitimate transactions incorrectly flagged
        false_negatives: missed fraud
        true_negatives:  correctly approved legitimate transactions
        avg_fraud_amount: average dollar value of a fraudulent transaction
        cost_model: configurable cost parameters

    Returns:
        Dict with individual and total cost breakdowns.
    """
    cm = cost_model or DEFAULT_CONFIG.cost_model

    # Costs incurred
    fp_review_cost = false_positives * cm.manual_review_cost
    fp_blocked_cost = false_positives * cm.blocked_legitimate_transaction_cost
    total_false_positive_cost = fp_review_cost + fp_blocked_cost

    total_fraud_loss = false_negatives * cm.fraud_loss_cost

    # Savings from catching fraud
    fraud_prevented = true_positives * avg_fraud_amount

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "fp_review_cost": round(fp_review_cost, 2),
        "fp_blocked_cost": round(fp_blocked_cost, 2),
        "total_false_positive_cost": round(total_false_positive_cost, 2),
        "total_fraud_loss": round(total_fraud_loss, 2),
        "fraud_prevented_value": round(fraud_prevented, 2),
        "net_cost": round(total_false_positive_cost + total_fraud_loss - fraud_prevented, 2),
    }
