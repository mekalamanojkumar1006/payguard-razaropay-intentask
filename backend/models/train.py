"""
PayGuard Model Training & Serialization.

Trains the baseline model, saves the pipeline and metadata to
backend/models/artifacts/ so the API can load it without retraining.
"""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from backend.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

TARGET = "is_fraud"
DROP_COLS = ["transaction_id", "customer_id", "timestamp", "device_id", "ip_address", TARGET]


def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, val, test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def evaluate(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "confusion_matrix": cm.tolist(),
    }


def train_and_save(
    model_type: str = "rf",
    data_dir: str | None = None,
    artifact_dir: str | None = None,
):
    """Train, evaluate, and persist the model pipeline."""
    data_dir = data_dir or DEFAULT_CONFIG.data_dir
    artifact_dir = artifact_dir or DEFAULT_CONFIG.model_artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)

    # Load data
    train, val, test = load_data(data_dir)

    X_train = train.drop(columns=DROP_COLS)
    y_train = train[TARGET]
    X_val = val.drop(columns=DROP_COLS)
    y_val = val[TARGET]
    X_test = test.drop(columns=DROP_COLS)
    y_test = test[TARGET]

    feature_columns = X_train.columns.tolist()

    # Build model pipeline
    preprocessor = build_preprocessor(X_train)
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(random_state=42, max_iter=1000)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    print(f"Training {model_type.upper()}...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    val_metrics = evaluate(y_val, pipeline.predict(X_val))
    test_metrics = evaluate(y_test, pipeline.predict(X_test))

    print(f"Validation: {val_metrics}")
    print(f"Test:       {test_metrics}")

    # Save artifacts
    model_path = os.path.join(artifact_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_type": model_type,
        "feature_columns": feature_columns,
        "trained_at": datetime.utcnow().isoformat(),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "random_seed": 42,
    }
    meta_path = os.path.join(artifact_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    return pipeline, feature_columns, metadata


def load_model(artifact_dir: str | None = None):
    """Load a previously trained model and its metadata."""
    artifact_dir = artifact_dir or DEFAULT_CONFIG.model_artifact_dir
    model_path = os.path.join(artifact_dir, "model.joblib")
    meta_path = os.path.join(artifact_dir, "metadata.json")

    pipeline = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    logger.info("Loaded model trained at %s", metadata.get("trained_at"))
    return pipeline, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and save PayGuard model")
    parser.add_argument("--model", default="rf", choices=["rf", "lr"])
    parser.add_argument("--data_dir", default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--artifact_dir", default=DEFAULT_CONFIG.model_artifact_dir)
    args = parser.parse_args()

    train_and_save(model_type=args.model, data_dir=args.data_dir, artifact_dir=args.artifact_dir)
