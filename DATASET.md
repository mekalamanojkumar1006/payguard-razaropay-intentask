# PAYGUARD DATASET

## Overview
This synthetic dataset simulates payment transactions for the PAYGUARD risk & fraud detection agent. 
It contains 10,000 transactions generated with realistic fraud patterns.

## Features
- `transaction_id`: Unique UUID for the transaction
- `customer_id`: Unique UUID for the customer
- `timestamp`: ISO 8601 timestamp
- `amount`: Transaction amount
- `currency`: USD
- `payment_method`: Method used (Credit Card, Debit Card, PayPal, Crypto)
- `merchant_category`: Retail, Travel, Digital Goods, Services, Groceries
- `device_id`: Device identifier
- `ip_address`: IPv4 address (anonymized)
- `country`: Country code (US, UK, CA, FR, DE, JP, IN)
- `customer_age_of_account`: Days since account creation
- `historical_average_amount`: Average amount historically spent by this customer
- `transactions_last_10_minutes`: Number of transactions by customer in last 10 mins
- `transactions_last_1_hour`: Number of transactions by customer in last 1 hour
- `transactions_last_24_hours`: Number of transactions by customer in last 24 hours
- `device_changed_recently`: Binary flag if device changed recently
- `location_changed_recently`: Binary flag if location changed recently
- `previous_chargebacks`: Number of historical chargebacks
- `previous_failed_transactions`: Number of recently failed transactions
- `account_velocity`: Ratio of 24h transactions to historical average amount
- `is_fraud`: Target variable (1 for fraud, 0 for legitimate)

## Fraud Patterns
1. High amount compared to historical average.
2. Device changes paired with transaction.
3. Sudden location changes.
4. Elevated short-term transaction velocity.
5. High previous failure / chargeback history.

## Data Splits
The data is sorted chronologically and split:
- **Train (70%)**: Initial learning data
- **Validation (15%)**: Hyperparameter tuning and model selection
- **Test (15%)**: Held-out final evaluation data (no leakage allowed)
