"""
PayGuard API -- FastAPI Application (Day 3).

Endpoints:
  GET  /health                  -- health check
  POST /api/v1/risk/analyze     -- quick risk analysis
  POST /api/v1/risk/investigate -- full investigation workflow
  GET  /api/v1/transactions     -- paginated transaction list
  GET  /api/v1/stats            -- dashboard aggregate stats
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import DEFAULT_CONFIG
from backend.models.train import load_model
from backend.services.orchestrator import PayGuardOrchestrator
from backend.services.customer_history import CustomerHistoryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """Validated transaction input."""
    transaction_id: str = Field(..., min_length=1)
    customer_id: str = Field(default="unknown")
    timestamp: str = Field(default="")
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD")
    payment_method: str = Field(default="Credit Card")
    merchant_category: str = Field(default="Retail")
    device_id: str = Field(default="unknown")
    ip_address: str = Field(default="0.0.0.0")
    country: str = Field(default="US")
    customer_age_of_account: int = Field(default=365, ge=0)
    historical_average_amount: float = Field(default=50.0, ge=0)
    transactions_last_10_minutes: int = Field(default=0, ge=0)
    transactions_last_1_hour: int = Field(default=0, ge=0)
    transactions_last_24_hours: int = Field(default=0, ge=0)
    device_changed_recently: int = Field(default=0, ge=0, le=1)
    location_changed_recently: int = Field(default=0, ge=0, le=1)
    previous_chargebacks: int = Field(default=0, ge=0)
    previous_failed_transactions: int = Field(default=0, ge=0)
    account_velocity: float = Field(default=0.0, ge=0)


class FindingResponse(BaseModel):
    reason_code: str
    category: str
    title: str
    severity: str
    evidence: str
    source_feature: str


class InvestigationResponse(BaseModel):
    summary: str
    risk_score: float
    risk_level: str
    findings: List[FindingResponse] = []
    behavioral_assessment: str = ""
    recommended_action: str
    key_reasons: List[str] = []
    evidence: List[str] = []


class RiskAnalysisResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    ml_probability: float
    decision: str
    risk_signals: list
    investigation: InvestigationResponse


class CustomerProfileResponse(BaseModel):
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


class InvestigateResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    ml_probability: float
    decision: str
    risk_signals: list
    investigation: InvestigationResponse
    customer_profile: Optional[CustomerProfileResponse] = None


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

orchestrator: Optional[PayGuardOrchestrator] = None
history_service: Optional[CustomerHistoryService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and customer history at startup."""
    global orchestrator, history_service

    # Customer history
    history_service = CustomerHistoryService()
    history_service.load()

    # Model
    try:
        pipeline, metadata = load_model()
        feature_columns = metadata["feature_columns"]
        llm_key = os.environ.get("LLM_API_KEY")
        orchestrator = PayGuardOrchestrator(
            model=pipeline,
            feature_columns=feature_columns,
            llm_api_key=llm_key,
        )
        logger.info("Orchestrator ready -- model trained at %s", metadata.get("trained_at"))
    except FileNotFoundError:
        logger.warning(
            "No saved model found at %s. "
            "Run `python -c \"from backend.models.train import train_and_save; train_and_save()\"` first.",
            DEFAULT_CONFIG.model_artifact_dir,
        )
    yield


app = FastAPI(title="PayGuard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_orchestrator():
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "payguard-backend",
        "model_loaded": orchestrator is not None,
    }


@app.post("/api/v1/risk/analyze", response_model=RiskAnalysisResponse)
def analyze_risk(tx: TransactionRequest):
    """Quick risk analysis pipeline."""
    _require_orchestrator()
    result = orchestrator.analyze(tx.model_dump())
    return result.to_dict()


@app.post("/api/v1/risk/investigate", response_model=InvestigateResponse)
def investigate_risk(tx: TransactionRequest):
    """Full investigation workflow with customer history."""
    _require_orchestrator()
    tx_dict = tx.model_dump()
    result = orchestrator.analyze(tx_dict)

    # Customer profile
    profile = None
    if history_service:
        p = history_service.get_profile(tx_dict.get("customer_id", ""))
        if p:
            profile = p.to_dict()
            # remove recent_transactions list from the profile response
            profile.pop("recent_transactions", None)

    resp = result.to_dict()
    resp["customer_profile"] = profile
    return resp


@app.get("/api/v1/transactions")
def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Paginated transaction list for the dashboard."""
    if history_service is None:
        raise HTTPException(status_code=503, detail="History service not loaded.")
    txns = history_service.get_transactions(limit=limit, offset=offset)
    total = history_service.get_total_count()
    return {"transactions": txns, "total": total, "limit": limit, "offset": offset}


@app.get("/api/v1/stats")
def dashboard_stats():
    """Aggregate stats for the dashboard overview cards."""
    if history_service is None:
        return {"total": 0, "customers": 0, "fraud_count": 0, "fraud_rate": 0}
    stats = history_service.get_stats()

    # Add model metrics if available
    model_metrics = None
    try:
        _, metadata = load_model()
        model_metrics = metadata.get("test_metrics")
    except Exception:
        pass

    stats["model_metrics"] = model_metrics
    return stats
