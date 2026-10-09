import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_elasticnet_baseline():
    print("[*] Loading raw biological matrices...")
    meth_df = pd.read_csv("data/processed/clean_meth.csv", index_col=0)
    ent_df = pd.read_csv("data/processed/clean_entropy.csv", index_col=0)
    ages_df = pd.read_csv("data/processed/clean_ages.csv", index_col=0)

    # Align patients exactly as load_raw_real_data does
    common_patients = meth_df.index.intersection(ent_df.index).intersection(ages_df.index)
    meth_df = meth_df.loc[common_patients]
    ent_df = ent_df.loc[common_patients]
    y = ages_df.loc[common_patients, 'Age'].values
    
    # Concatenate features to match the dual-stream input
    X = pd.concat([meth_df, ent_df], axis=1).values
    
    # EXACT STRATIFIED SPLIT FROM run_real.py
    indices = np.arange(len(meth_df))
    age_bins = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))
    
    # First split: 70% Train, 30% Temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=age_bins)
    
    # Second split: 15% Val, 15% Test
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=age_bins[temp_idx])
    
    print(f"[*] Matched Split sizes -> train: {len(train_idx)}, test: {len(test_idx)}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Scale features using StandardScaler, fitting only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("[*] Training ElasticNetCV baseline...")
    baseline_model = ElasticNetCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
    baseline_model.fit(X_train_scaled, y_train)
    
    print("[*] Evaluating baseline on exact matched test split...")
    predictions = baseline_model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*40)
    print(" BASELINE PERFORMANCE (ElasticNet)")
    print("="*40)
    print(f" Mean Absolute Error (MAE) : {mae:.2f} years")
    print(f" Root Mean Squared (RMSE)  : {rmse:.2f} years")
    print("="*40)

if __name__ == "__main__":
    run_elasticnet_baseline()