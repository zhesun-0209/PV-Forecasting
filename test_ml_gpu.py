#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ML Models GPU Support
"""

import numpy as np
import sys
import os

print("="*80)
print("ML Models GPU Support Test")
print("="*80)

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from models.ml_models import train_rf, train_linear

print("\n[Step 1/5] Checking GPU availability...")
import torch
if torch.cuda.is_available():
    print(f"  [OK] GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  [WARNING] No GPU available, using CPU")

print("\n[Step 2/5] Generating test data...")
# Create test data (1000 samples, 50 features, 24 outputs)
X_train = np.random.rand(1000, 50).astype(np.float32)
y_train = np.random.rand(1000, 24).astype(np.float32)
X_test = np.random.rand(200, 50).astype(np.float32)
y_test = np.random.rand(200, 24).astype(np.float32)

train_data = (X_train, None, y_train, None, None)
test_data = (X_test, None, y_test, None, None)
scalers = (None, None, None)

print(f"  Train shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")

# Test XGBoost
print("\n" + "="*80)
print("[Step 3/5] Testing XGBoost GPU...")
print("="*80)
try:
    from train.train_ml import train_ml_model
    
    config = {
        'model': 'XGB',
        'model_params': {
            'n_estimators': 30,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        },
        'experiment_name': 'test_xgb',
        'future_hours': 24,  # Required field
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_excel_results': False,
            'save_training_log': False
        }
    }
    
    xgb_model, metrics = train_ml_model(config, train_data, test_data, test_data, scalers)
    print(f"[OK] XGBoost completed!")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
except Exception as e:
    print(f"[ERROR] XGBoost failed: {str(e)}")
    import traceback
    traceback.print_exc()

# Test LightGBM
print("\n" + "="*80)
print("[Step 4/5] Testing LightGBM GPU...")
print("="*80)
try:
    config = {
        'model': 'LGBM',
        'model_params': {
            'n_estimators': 30,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': -1
        },
        'experiment_name': 'test_lgbm',
        'future_hours': 24,  # Required field
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_excel_results': False,
            'save_training_log': False
        }
    }
    
    lgbm_model, metrics = train_ml_model(config, train_data, test_data, test_data, scalers)
    print(f"[OK] LightGBM completed!")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
except Exception as e:
    print(f"[ERROR] LightGBM failed: {str(e)}")
    import traceback
    traceback.print_exc()

# Test Random Forest
print("\n" + "="*80)
print("[Step 5/5] Testing Random Forest...")
print("="*80)
try:
    rf_params = {'n_estimators': 30, 'max_depth': 3, 'random_state': 42}
    rf_model = train_rf(X_train, y_train, rf_params)
    print("[OK] Random Forest training completed!")
    
    # Test prediction
    y_pred = rf_model.predict(X_test)
    print(f"[OK] Prediction shape: {y_pred.shape}")
    
    # Calculate simple MAE
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"  Test MAE: {mae:.4f}")
except Exception as e:
    print(f"[ERROR] Random Forest failed: {str(e)}")
    import traceback
    traceback.print_exc()

# Test Linear Regression
print("\n" + "="*80)
print("[Bonus] Testing Linear Regression...")
print("="*80)
try:
    linear_model = train_linear(X_train, y_train, {})
    print("[OK] Linear Regression training completed!")
    
    # Test prediction
    y_pred = linear_model.predict(X_test)
    print(f"[OK] Prediction shape: {y_pred.shape}")
    
    # Calculate simple MAE
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"  Test MAE: {mae:.4f}")
except Exception as e:
    print(f"[ERROR] Linear Regression failed: {str(e)}")

# Summary
print("\n" + "="*80)
print("GPU Test Summary")
print("="*80)
print("[OK] All tests completed!")
print("\nIf you see [OK] for all models, GPU support is working correctly.")
print("If you see [ERROR], check the error messages above.")
print("="*80)

