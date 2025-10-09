#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Plant Experiment Runner - Batch run 284 experiments for multiple plants
Uses unified configuration system for easy management and reusability
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from datetime import datetime
from typing import List, Dict

warnings.filterwarnings('ignore')

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from config_manager import PlantConfigManager
from data.data_utils import preprocess_features, create_daily_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model


def run_single_experiment(config: Dict, df: pd.DataFrame) -> Dict:
    """
    Run a single experiment
    
    Args:
        config: Experiment configuration
        df: Data DataFrame
        
    Returns:
        Experiment result dictionary
    """
    try:
        # Data preprocessing
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
        
        # Create daily windows (one prediction per day)
        # Aligns with day-ahead forecasting scenario
        # Supports variable lookback (24h=1day, 72h=3days)
        past_hours = config.get('past_hours', 24)
        X_hist, X_fcst, y, hours, dates = create_daily_windows(
            df_clean, config['future_hours'], hist_feats, fcst_feats, no_hist_power, past_hours
        )
        
        # Data splitting: Random shuffle for robust evaluation
        # Covers all seasons and ensures model generalization
        total_samples = len(X_hist)
        indices = np.arange(total_samples)
        
        shuffle_split = config.get('shuffle_split', True)
        random_seed = config.get('random_seed', 42)
        
        if shuffle_split:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        print(f"  Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        print(f"  Test period: {dates[test_idx[0]]} to {dates[test_idx[-1]]}")
        
        X_hist_train, y_train = X_hist[train_idx], y[train_idx]
        X_hist_val, y_val = X_hist[val_idx], y[val_idx]
        X_hist_test, y_test = X_hist[test_idx], y[test_idx]
        
        if X_fcst is not None:
            X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
        else:
            X_fcst_train = X_fcst_val = X_fcst_test = None
        
        # Split hours and dates
        train_hours = np.array([hours[i] for i in train_idx])
        val_hours = np.array([hours[i] for i in val_idx])
        test_hours = np.array([hours[i] for i in test_idx])
        test_dates = [dates[i] for i in test_idx]
        
        train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
        val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
        test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
        scalers = (scaler_hist, scaler_fcst, scaler_target)
        
        # Train model
        start_time = time.time()
        if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
        else:
            model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
        training_time = time.time() - start_time
        
        # Parse scenario name
        use_pv = config.get('use_pv', False)
        use_hist_weather = config.get('use_hist_weather', False)
        use_forecast = config.get('use_forecast', False)
        use_ideal_nwp = config.get('use_ideal_nwp', False)
        
        if use_pv and use_hist_weather:
            scenario = 'PV+HW'
        elif use_pv and use_forecast and use_ideal_nwp:
            scenario = 'PV+NWP+'
        elif use_pv and use_forecast:
            scenario = 'PV+NWP'
        elif use_pv:
            scenario = 'PV'
        elif use_forecast and use_ideal_nwp:
            scenario = 'NWP+'
        elif use_forecast:
            scenario = 'NWP'
        else:
            scenario = 'Unknown'
        
        # Return result
        result = {
            'plant_id': config['plant_id'],
            'experiment_name': config['experiment_name'],
            'model': config['model'],
            'complexity': config.get('model_complexity', 'N/A'),
            'scenario': scenario,
            'lookback_hours': config['past_hours'],
            'use_time_encoding': config['use_time_encoding'],
            'mae': metrics.get('mae', 0.0),
            'rmse': metrics.get('rmse', 0.0),
            'r2': metrics.get('r2', 0.0),
            'train_time_sec': round(training_time, 2),
            'test_samples': metrics.get('samples_count', 0),
            'best_epoch': int(metrics.get('best_epoch', 0)) if not pd.isna(metrics.get('best_epoch', 0)) else 0,
            'param_count': int(metrics.get('param_count', 0)),
            'status': 'SUCCESS'
        }
        
        return result
        
    except Exception as e:
        print(f"  [ERROR] Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'plant_id': config.get('plant_id', 'Unknown'),
            'experiment_name': config.get('experiment_name', 'Unknown'),
            'model': config.get('model', 'Unknown'),
            'complexity': config.get('model_complexity', 'N/A'),
            'scenario': 'FAILED',
            'lookback_hours': config.get('past_hours', 0),
            'use_time_encoding': config.get('use_time_encoding', False),
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'train_time_sec': 0,
            'test_samples': 0,
            'best_epoch': 0,
            'param_count': 0,
            'status': 'FAILED',
            'error': str(e)
        }


def run_plant_experiments(plant_config_path: str, resume: bool = True):
    """
    Run all experiments for a single plant
    
    Args:
        plant_config_path: Plant configuration file path
        resume: Whether to support resume from checkpoint
        
    Returns:
        Number of successful experiments
    """
    print("\n" + "=" * 80)
    print(f"Running experiments for plant: {plant_config_path}")
    print("=" * 80)
    
    # Load configuration
    manager = PlantConfigManager()
    plant_config = manager.load_plant_config(plant_config_path)
    plant_id = plant_config['plant_id']
    
    # Generate 284 experiment configurations
    all_configs = manager.generate_experiment_configs(plant_config)
    print(f"Total configurations generated: {len(all_configs)}")
    
    # Load data
    data_path = plant_config['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return 0
    
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    # GPU information
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check for existing results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{plant_id}_{timestamp}.csv"
    done_experiments = set()
    
    if resume:
        # Find the latest result file
        existing_files = [f for f in os.listdir(script_dir)
                         if f.startswith(f"results_{plant_id}_") and f.endswith(".csv")]
        if existing_files:
            existing_files.sort(key=lambda x: os.path.getmtime(os.path.join(script_dir, x)), reverse=True)
            output_file = existing_files[0]
            print(f"Found existing result file: {output_file}")
            results_df = pd.read_csv(output_file)
            
            # Fix: Check if 'status' column exists to avoid resume bug
            if 'status' in results_df.columns:
                done_experiments = set(results_df[results_df['status'] == 'SUCCESS']["experiment_name"].tolist())
            else:
                # If no status column, assume all experiments in file are completed
                done_experiments = set(results_df["experiment_name"].dropna().tolist())
            print(f"Already completed: {len(done_experiments)}")
    
    if not done_experiments:
        # Create new file
        results_df = pd.DataFrame(columns=[
            'plant_id', 'experiment_name', 'model', 'complexity', 'scenario',
            'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2',
            'train_time_sec', 'test_samples', 'best_epoch', 'param_count', 'status'
        ])
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Remaining: {len(all_configs) - len(done_experiments)}")
    
    # Run experiments
    success_count = 0
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        
        if exp_name in done_experiments:
            print(f"[{idx}/{len(all_configs)}] SKIP: {exp_name} (already completed)")
            success_count += 1
            continue
        
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(all_configs)}] Running: {exp_name}")
        print(f"{'=' * 80}")
        
        # Run experiment
        result = run_single_experiment(config, df)
        
        # Save result
        pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        if result['status'] == 'SUCCESS':
            print(f"  [OK] MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
            success_count += 1
        else:
            print(f"  [FAILED] {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 80}")
    print(f"Plant {plant_id} Experiments Completed!")
    print(f"Success: {success_count}/{len(all_configs)}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}\n")
    
    return success_count


def run_all_plants(resume: bool = True):
    """
    Run experiments for all plants
    
    Args:
        resume: Whether to support resume from checkpoint
    """
    print("=" * 80)
    print("Running experiments for all plants")
    print("=" * 80)
    
    # Get all plant configurations
    manager = PlantConfigManager()
    plants = manager.get_all_plants()
    
    if not plants:
        print("No plant configurations found in config/plants/")
        print("Please create plant config files first.")
        print("Example: python config_manager.py create 1140 data/Project1140.csv 2022-01-01 2024-09-28")
        return
    
    print(f"Found {len(plants)} plants:")
    for plant in plants:
        print(f"  - Plant {plant['plant_id']}: {plant['data_path']}")
    print()
    
    # Run each plant sequentially
    total_success = 0
    total_experiments = 0
    
    for i, plant in enumerate(plants, 1):
        plant_id = plant['plant_id']
        plant_config_path = f"config/plants/Plant{plant_id}.yaml"
        
        print(f"\n{'#' * 80}")
        print(f"Plant {i}/{len(plants)}: {plant_id}")
        print(f"{'#' * 80}")
        
        success = run_plant_experiments(plant_config_path, resume=resume)
        total_success += success
        total_experiments += 284  # Each plant has 284 experiments
    
    print("\n" + "=" * 80)
    print("ALL PLANTS COMPLETED!")
    print(f"Total Success: {total_success}/{total_experiments}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-plant PV forecasting experiments')
    parser.add_argument('--plant', type=str, help='Plant ID to run (e.g., 1140). If not specified, run all plants.')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh without resuming from previous results')
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    
    if args.plant:
        # Run specified plant
        plant_config_path = f"config/plants/Plant{args.plant}.yaml"
        if not os.path.exists(plant_config_path):
            print(f"Error: Plant config not found: {plant_config_path}")
            print(f"Available plants:")
            manager = PlantConfigManager()
            plants = manager.get_all_plants()
            for plant in plants:
                print(f"  - Plant {plant['plant_id']}")
            sys.exit(1)
        
        run_plant_experiments(plant_config_path, resume=resume)
    else:
        # Run all plants
        run_all_plants(resume=resume)