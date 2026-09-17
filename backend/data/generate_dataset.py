import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import os
import argparse

def generate_transactions(n_samples=10000, fraud_ratio=0.05, random_seed=42):
    np.random.seed(random_seed)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    customers = [str(uuid.uuid4()) for _ in range(1000)]
    devices = [str(uuid.uuid4()) for _ in range(1200)]
    merchants = ["Retail", "Travel", "Digital Goods", "Services", "Groceries"]
    countries = ["US", "UK", "CA", "FR", "DE", "JP", "IN"]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        is_fraud = 1 if i < n_fraud else 0
        
        # Base features
        cust_id = np.random.choice(customers)
        dev_id = np.random.choice(devices)
        merchant = np.random.choice(merchants)
        country = np.random.choice(countries)
        
        # Continuous / Time
        ts = start_date + timedelta(days=np.random.randint(0, 180), minutes=np.random.randint(0, 1440))
        
        # Behavioral Features
        if is_fraud:
            # Add some fraud patterns
            pattern = np.random.randint(1, 6)
            if pattern == 1:
                amt = np.random.exponential(500) + 500  # High amount
                hist_avg = np.random.exponential(50) + 10
            elif pattern == 2:
                amt = np.random.exponential(50) + 10
                hist_avg = amt
                dev_id = np.random.choice(devices) # Dev changed
            else:
                amt = np.random.exponential(100) + 20
                hist_avg = amt
        else:
            amt = np.random.exponential(50) + 10
            hist_avg = amt + np.random.normal(0, 10)
            hist_avg = max(5, hist_avg)
            
        amount = round(amt, 2)
        historical_average_amount = round(hist_avg, 2)
        
        # Signal injection
        device_changed_recently = 1 if (is_fraud and np.random.random() < 0.6) else (1 if np.random.random() < 0.1 else 0)
        location_changed_recently = 1 if (is_fraud and np.random.random() < 0.5) else (1 if np.random.random() < 0.05 else 0)
        
        previous_chargebacks = np.random.randint(1, 5) if (is_fraud and np.random.random() < 0.3) else 0
        previous_failed_transactions = np.random.randint(1, 10) if (is_fraud and np.random.random() < 0.4) else np.random.randint(0, 2)
        
        tx_10m = np.random.randint(2, 12) if (is_fraud and np.random.random() < 0.7) else np.random.randint(0, 5)
        tx_1h = tx_10m + np.random.randint(0, 5) if (is_fraud and np.random.random() < 0.6) else tx_10m + np.random.randint(0, 3)
        tx_24h = tx_1h + np.random.randint(0, 10) if (is_fraud and np.random.random() < 0.5) else tx_1h + np.random.randint(0, 6)
        
        account_velocity = tx_24h / (historical_average_amount + 1)
        
        row = {
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cust_id,
            "timestamp": ts.isoformat(),
            "amount": amount,
            "currency": "USD",
            "payment_method": np.random.choice(["Credit Card", "Debit Card", "PayPal", "Crypto"]),
            "merchant_category": merchant,
            "device_id": dev_id,
            "ip_address": f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.0.0",
            "country": country,
            "customer_age_of_account": np.random.randint(1, 3650),
            "historical_average_amount": historical_average_amount,
            "transactions_last_10_minutes": tx_10m,
            "transactions_last_1_hour": tx_1h,
            "transactions_last_24_hours": tx_24h,
            "device_changed_recently": device_changed_recently,
            "location_changed_recently": location_changed_recently,
            "previous_chargebacks": previous_chargebacks,
            "previous_failed_transactions": previous_failed_transactions,
            "account_velocity": round(account_velocity, 4),
            "is_fraud": is_fraud
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df

def save_splits(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Time-aware split: first 70% train, next 15% val, last 15% test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Generated {n} total transactions.")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Overall Fraud Ratio: {df['is_fraud'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="backend/data")
    args = parser.parse_args()
    
    df = generate_transactions(n_samples=args.samples)
    save_splits(df, args.output)
