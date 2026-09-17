"""
PayGuard Customer History Service.

Provides lightweight in-memory lookup for customer transaction history
from the synthetic dataset.  Builds a customer-keyed index once at
startup so individual lookups are O(1) hash + small-list scan.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import pandas as pd

from backend.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CustomerProfile:
    """Aggregated customer profile from historical transactions."""
    customer_id: str
    total_transactions: int
    historical_average_amount: float
    max_amount: float
    recent_transaction_count_24h: int
    distinct_devices: int
    distinct_countries: int
    total_chargebacks: int
    total_failed_transactions: int
    account_age_days: int
    recent_transactions: List[Dict[str, Any]]  # last 10 transactions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CustomerHistoryService:
    """
    In-memory customer history index.

    Loads the full training dataset once and indexes by customer_id.
    For a demo/internship project this is adequate; in production
    this would be backed by a database.
    """

    def __init__(self, data_dir: str | None = None):
        self._data_dir = data_dir or DEFAULT_CONFIG.data_dir
        self._index: Dict[str, List[Dict[str, Any]]] = {}
        self._all_transactions: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        """Load and index the dataset."""
        if self._loaded:
            return

        frames = []
        for split in ["train.csv", "val.csv", "test.csv"]:
            path = os.path.join(self._data_dir, split)
            if os.path.exists(path):
                frames.append(pd.read_csv(path))

        if not frames:
            logger.warning("No data files found in %s", self._data_dir)
            self._loaded = True
            return

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._all_transactions = df.to_dict("records")

        # Build customer index
        for tx in self._all_transactions:
            cid = tx.get("customer_id", "unknown")
            self._index.setdefault(cid, []).append(tx)

        logger.info(
            "Customer history loaded: %d transactions, %d customers",
            len(self._all_transactions), len(self._index),
        )
        self._loaded = True

    def get_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Return an aggregated profile for a customer."""
        txns = self._index.get(customer_id)
        if not txns:
            return None

        amounts = [t["amount"] for t in txns]
        devices = set(t.get("device_id", "") for t in txns)
        countries = set(t.get("country", "") for t in txns)

        return CustomerProfile(
            customer_id=customer_id,
            total_transactions=len(txns),
            historical_average_amount=round(sum(amounts) / len(amounts), 2),
            max_amount=max(amounts),
            recent_transaction_count_24h=max(
                (t.get("transactions_last_24_hours", 0) for t in txns), default=0
            ),
            distinct_devices=len(devices),
            distinct_countries=len(countries),
            total_chargebacks=sum(t.get("previous_chargebacks", 0) for t in txns),
            total_failed_transactions=sum(
                t.get("previous_failed_transactions", 0) for t in txns
            ),
            account_age_days=max(
                (t.get("customer_age_of_account", 0) for t in txns), default=0
            ),
            recent_transactions=[
                _slim_tx(t) for t in txns[-10:]
            ],
        )

    def get_transactions(
        self, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Return a paginated slice of all transactions (most-recent first)."""
        reversed_txns = list(reversed(self._all_transactions))
        return [_slim_tx(t) for t in reversed_txns[offset : offset + limit]]

    def get_total_count(self) -> int:
        return len(self._all_transactions)

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate dataset statistics for the dashboard."""
        if not self._all_transactions:
            return {"total": 0, "fraud_count": 0, "fraud_rate": 0}
        total = len(self._all_transactions)
        fraud_count = sum(1 for t in self._all_transactions if t.get("is_fraud") == 1)
        return {
            "total": total,
            "customers": len(self._index),
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / total, 4) if total else 0,
        }


def _slim_tx(t: Dict[str, Any]) -> Dict[str, Any]:
    """Return a frontend-safe subset of a transaction dict."""
    return {
        "transaction_id": t.get("transaction_id"),
        "customer_id": t.get("customer_id"),
        "timestamp": t.get("timestamp"),
        "amount": t.get("amount"),
        "currency": t.get("currency", "USD"),
        "payment_method": t.get("payment_method"),
        "merchant_category": t.get("merchant_category"),
        "country": t.get("country"),
        "device_changed_recently": t.get("device_changed_recently", 0),
        "location_changed_recently": t.get("location_changed_recently", 0),
        "is_fraud": t.get("is_fraud"),
    }
