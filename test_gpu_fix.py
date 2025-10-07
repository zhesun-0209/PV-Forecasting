#!/usr/bin/env python3
"""
测试修复后的GPU支持
"""

import numpy as np
import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from models.ml_models import train_rf, train_linear

def test_gpu_support():
    """Test GPU support after fix"""
    print("Testing GPU support after fix")
    print("=" * 50)
    
    # Create test data
    print("Creating test data...")
    X_train = np.random.rand(1000, 50).astype(np.float32)
    y_train = np.random.rand(1000, 24).astype(np.float32)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Test RF GPU
    print("\nTesting Random Forest GPU...")
    try:
        rf_params = {
            'n_estimators': 50,
            'max_depth': 8,
            'random_state': 42
        }
        rf_model = train_rf(X_train, y_train, rf_params)
        
        # Test prediction
        X_test = np.random.rand(100, 50).astype(np.float32)
        y_pred = rf_model.predict(X_test)
        print(f"SUCCESS: RF training completed! Prediction shape: {y_pred.shape}")
        
    except Exception as e:
        print(f"FAILED: RF training failed: {e}")
    
    # Test Linear GPU
    print("\nTesting Linear Regression GPU...")
    try:
        linear_params = {}
        linear_model = train_linear(X_train, y_train, linear_params)
        
        # Test prediction
        X_test = np.random.rand(100, 50).astype(np.float32)
        y_pred = linear_model.predict(X_test)
        print(f"SUCCESS: Linear training completed! Prediction shape: {y_pred.shape}")
        
    except Exception as e:
        print(f"FAILED: Linear training failed: {e}")
    
    print("\nGPU testing completed!")

if __name__ == "__main__":
    test_gpu_support()
