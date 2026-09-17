import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import os
import argparse
import numpy as np

def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, val, test

def prepare_data(train, val, test):
    target = "is_fraud"
    
    # Drop identifying or datetime columns for baseline
    drop_cols = ["transaction_id", "customer_id", "timestamp", "device_id", "ip_address", target]
    
    X_train = train.drop(columns=drop_cols)
    y_train = train[target]
    
    X_val = val.drop(columns=drop_cols)
    y_val = val[target]
    
    X_test = test.drop(columns=drop_cols)
    y_test = test[target]
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
            
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor

def evaluate(y_true, y_pred, name="Validation"):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"--- {name} Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    return precision, recall, f1, fpr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="backend/data")
    parser.add_argument("--model", type=str, default="rf", choices=["lr", "rf"])
    args = parser.parse_args()
    
    train, val, test = load_data(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = prepare_data(train, val, test)
    
    if args.model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(random_state=42, max_iter=1000)
        
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', clf)])
                            
    print(f"Training {args.model.upper()}...")
    model.fit(X_train, y_train)
    
    # Validate
    y_val_pred = model.predict(X_val)
    evaluate(y_val, y_val_pred, "Validation")
    
    # Test
    y_test_pred = model.predict(X_test)
    evaluate(y_test, y_test_pred, "Test")

if __name__ == "__main__":
    main()
