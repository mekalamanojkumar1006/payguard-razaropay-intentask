import pytest
import pandas as pd
import os

DATA_DIR = "backend/data"

@pytest.fixture
def data_splits():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train, val, test

def test_missing_values(data_splits):
    train, val, test = data_splits
    assert train.isnull().sum().sum() == 0, "Missing values found in train set"
    assert val.isnull().sum().sum() == 0, "Missing values found in val set"
    assert test.isnull().sum().sum() == 0, "Missing values found in test set"

def test_duplicate_transaction_ids(data_splits):
    train, val, test = data_splits
    all_data = pd.concat([train, val, test])
    assert all_data["transaction_id"].is_unique, "Duplicate transaction IDs found across splits"

def test_invalid_amounts(data_splits):
    train, val, test = data_splits
    all_data = pd.concat([train, val, test])
    assert (all_data["amount"] < 0).sum() == 0, "Negative amounts found"

def test_leakage(data_splits):
    train, val, test = data_splits
    train_ids = set(train["transaction_id"])
    val_ids = set(val["transaction_id"])
    test_ids = set(test["transaction_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test"

def test_class_distribution(data_splits):
    train, val, test = data_splits
    train_fraud_ratio = train["is_fraud"].mean()
    assert 0.01 < train_fraud_ratio < 0.20, f"Unrealistic train fraud ratio: {train_fraud_ratio}"
