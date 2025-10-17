#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to diagnose sensitivity analysis issues
"""

import os
import sys

print("=" * 80)
print("SENSITIVITY ANALYSIS DIAGNOSTIC TEST")
print("=" * 80)

# Check 1: Data directory
print("\n1. Checking data directory...")
data_dir = 'data'
if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"   [OK] Found {len(csv_files)} CSV files")
    if len(csv_files) > 0:
        print(f"   Sample files: {csv_files[:3]}")
    else:
        print("   [ERROR] No CSV files found!")
else:
    print(f"   [ERROR] Data directory '{data_dir}' does not exist!")

# Check 2: Import modules
print("\n2. Checking module imports...")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from sensitivity_analysis.common_utils import load_all_plant_configs
    print("   [OK] common_utils imported")
except Exception as e:
    print(f"   [ERROR] Failed to import: {e}")
    sys.exit(1)

# Check 3: Load plant configs
print("\n3. Loading plant configurations...")
try:
    plant_configs = load_all_plant_configs(data_dir)
    print(f"   [OK] Loaded {len(plant_configs)} plant configurations")
    
    if len(plant_configs) == 0:
        print("   [ERROR] No plant configurations loaded!")
        print("   This is why experiments complete so quickly - no data to process!")
    else:
        # Show first plant config
        first_plant = plant_configs[0]
        print(f"   Sample plant ID: {first_plant['plant_id']}")
        print(f"   Sample data path: {first_plant['data_path']}")
        
        # Check if data file exists
        if os.path.exists(first_plant['data_path']):
            print(f"   [OK] Data file exists")
            
            # Try to load the data
            import pandas as pd
            df = pd.read_csv(first_plant['data_path'])
            print(f"   [OK] Data file loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {df.columns.tolist()[:5]}...")
        else:
            print(f"   [ERROR] Data file does not exist: {first_plant['data_path']}")
            
except Exception as e:
    print(f"   [ERROR] Failed to load configs: {e}")
    import traceback
    traceback.print_exc()

# Check 4: GPU availability
print("\n4. Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   [OK] GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("   [WARNING] No GPU detected (will use CPU)")
except ImportError:
    print("   [WARNING] PyTorch not installed")

# Check 5: Test single model training
print("\n5. Testing single model training (this may take a few minutes)...")
if len(plant_configs) > 0:
    try:
        from sensitivity_analysis.common_utils import create_base_config
        from data.data_utils import load_raw_data, preprocess_features
        from train.train_ml import train_ml_model
        
        # Use first plant
        plant_config = plant_configs[0]
        print(f"   Using plant: {plant_config['plant_id']}")
        
        # Create config for Linear model (fastest to test)
        config = create_base_config(plant_config, 'Linear', complexity='high', 
                                   lookback=24, use_te=False)
        config['use_pv'] = False
        config['use_hist_weather'] = False
        config['no_hist_power'] = True
        config['past_hours'] = 0
        
        # Load and preprocess data
        df = load_raw_data(plant_config['data_path'])
        print(f"   [OK] Loaded {len(df)} rows")
        
        df_processed = preprocess_features(df, config)
        print(f"   [OK] Preprocessed {len(df_processed)} rows")
        
        # Train model
        print("   Training Linear model (this should take 10-30 seconds)...")
        import time
        start_time = time.time()
        result = train_ml_model(config, df_processed)
        train_time = time.time() - start_time
        
        print(f"   [OK] Training completed in {train_time:.1f} seconds")
        print(f"   Test MAE: {result['test_mae']:.4f}")
        print(f"   Test RMSE: {result['test_rmse']:.4f}")
        print(f"   Test R2: {result['test_r2']:.4f}")
        
    except Exception as e:
        print(f"   [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   [SKIPPED] No plant configurations to test")

print("\n" + "=" * 80)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 80)
print("\nIf experiments are completing in < 2 minutes:")
print("  1. Check that CSV files exist in 'data/' directory")
print("  2. Check that plant configs are being loaded correctly")
print("  3. Check for errors in the experiment scripts")
print("  4. Run this test script to identify the issue")
print("\nExpected time for 1 plant, 1 model: ~30 seconds")
print("Expected time for 100 plants, 8 models: ~2-4 hours per experiment")
print("=" * 80)

