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


def check_plant_completion(plant_id: str, output_dir: str = None) -> tuple:
    """
    Check if a plant's experiments are complete
    
    Args:
        plant_id: Plant ID
        output_dir: Directory to check for results
        
    Returns:
        (is_complete, completed_count, result_file_path)
    """
    if output_dir is None:
        output_dir = script_dir
    
    # Check for result file for this plant
    if not os.path.exists(output_dir):
        return False, 0, None
    
    result_file = os.path.join(output_dir, f"results_{plant_id}.csv")
    
    if not os.path.exists(result_file):
        return False, 0, None
    
    try:
        df = pd.read_csv(result_file)
        
        # Count successful experiments
        if 'status' in df.columns:
            completed = len(df[df['status'] == 'SUCCESS'])
        else:
            # No status column, count non-null experiment names
            completed = len(df[df['experiment_name'].notna()])
        
        is_complete = (completed >= 284)
        return is_complete, completed, result_file
    
    except Exception as e:
        print(f"  Warning: Error reading {result_file}: {str(e)}")
        return False, 0, None


def run_plant_experiments(plant_config_path: str, resume: bool = True, output_dir: str = None):
    """
    Run all experiments for a single plant with resume support
    
    Args:
        plant_config_path: Plant configuration file path
        resume: Whether to support resume from checkpoint
        output_dir: Directory to save results (default: current directory)
        
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
    
    # Set output directory
    if output_dir is None:
        output_dir = script_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if already complete
    is_complete, completed_count, existing_file = check_plant_completion(plant_id, output_dir)
    
    if is_complete and resume:
        print(f"[OK] Plant {plant_id} already complete: {completed_count}/284 experiments")
        print(f"  Result file: {existing_file}")
        print(f"  Skipping to next plant...\n")
        return completed_count
    
    # Generate 284 experiment configurations
    all_configs = manager.generate_experiment_configs(plant_config)
    print(f"Total configurations: {len(all_configs)}")
    
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
    
    # Set output file path
    output_file = os.path.join(output_dir, f"results_{plant_id}.csv")
    
    # Resume from existing results
    done_experiments = set()
    
    if resume and existing_file:
        print(f"Resuming from: {output_file}")
        results_df = pd.read_csv(output_file)
        
        # Get completed experiments
        if 'status' in results_df.columns:
            done_experiments = set(results_df[results_df['status'] == 'SUCCESS']["experiment_name"].tolist())
        else:
            done_experiments = set(results_df["experiment_name"].dropna().tolist())
        
        print(f"Already completed: {len(done_experiments)}/284")
        print(f"Remaining: {len(all_configs) - len(done_experiments)}/284")
    else:
        # Create new file
        results_df = pd.DataFrame(columns=[
            'plant_id', 'experiment_name', 'model', 'complexity', 'scenario',
            'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2',
            'train_time_sec', 'test_samples', 'best_epoch', 'param_count', 'status'
        ])
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Created new result file: {output_file}")
    
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


def scan_all_plants_status(output_dir: str = None) -> List[Dict]:
    """
    Scan all plants and their completion status
    
    Args:
        output_dir: Directory to check for results
        
    Returns:
        List of plant status dictionaries
    """
    manager = PlantConfigManager()
    plants = manager.get_all_plants()
    
    plant_statuses = []
    
    for plant in plants:
        plant_id = plant['plant_id']
        is_complete, completed, result_file = check_plant_completion(plant_id, output_dir)
        
        plant_statuses.append({
            'plant_id': plant_id,
            'data_path': plant['data_path'],
            'is_complete': is_complete,
            'completed_experiments': completed,
            'remaining_experiments': 284 - completed,
            'result_file': result_file,
            'status': 'COMPLETE' if is_complete else ('IN_PROGRESS' if completed > 0 else 'NOT_STARTED')
        })
    
    return plant_statuses


def run_all_plants(resume: bool = True, skip: int = 0, max_plants: int = None, 
                   plants: List[str] = None, output_dir: str = None):
    """
    Run experiments for all plants with advanced filtering
    
    Args:
        resume: Whether to support resume from checkpoint
        skip: Number of plants to skip from the beginning
        max_plants: Maximum number of plants to process
        plants: List of specific plant IDs to run (overrides skip/max_plants)
        output_dir: Directory to save results (default: current directory)
    """
    print("=" * 80)
    print("Multi-Plant Batch Experiment Runner")
    print("=" * 80)
    
    # Get all plant configurations
    manager = PlantConfigManager()
    all_plants = manager.get_all_plants()
    
    if not all_plants:
        print("[ERROR] No plant configurations found in config/plants/")
        print("Please run: python batch_create_configs.py")
        return
    
    # Filter plants based on arguments
    if plants:
        # Run specific plants
        filtered_plants = [p for p in all_plants if p['plant_id'] in plants]
        print(f"Running specified plants: {plants}")
    else:
        # Apply skip and max_plants
        filtered_plants = all_plants[skip:]
        if max_plants:
            filtered_plants = filtered_plants[:max_plants]
        print(f"Total plants available: {len(all_plants)}")
        if skip > 0:
            print(f"Skipping first: {skip} plants")
        if max_plants:
            print(f"Running maximum: {max_plants} plants")
    
    # Set output directory
    if output_dir is None:
        output_dir = script_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\n{'='*80}")
    print(f"Plants to process: {len(filtered_plants)}")
    print(f"{'='*80}")
    
    # Scan status before starting
    if resume:
        print("\n[Scanning existing results...]")
        plant_statuses = []
        for plant in filtered_plants:
            is_complete, completed, result_file = check_plant_completion(plant['plant_id'], output_dir)
            plant_statuses.append({
                'plant_id': plant['plant_id'],
                'completed': completed,
                'is_complete': is_complete
            })
        
        complete_count = sum(1 for s in plant_statuses if s['is_complete'])
        in_progress_count = sum(1 for s in plant_statuses if 0 < s['completed'] < 284)
        not_started_count = sum(1 for s in plant_statuses if s['completed'] == 0)
        
        print(f"\nStatus Summary:")
        print(f"  [COMPLETE]:     {complete_count} plants")
        print(f"  [IN_PROGRESS]:  {in_progress_count} plants")
        print(f"  [NOT_STARTED]:  {not_started_count} plants")
        print(f"  [TO_RUN]:       {len(filtered_plants) - complete_count} plants")
    
    print(f"\n{'='*80}\n")
    
    # Run each plant sequentially
    total_success = 0
    total_experiments = 0
    plants_processed = 0
    
    start_time = time.time()
    
    for i, plant in enumerate(filtered_plants, 1):
        plant_id = plant['plant_id']
        plant_config_path = f"config/plants/Plant{plant_id}.yaml"
        
        print(f"\n{'#' * 80}")
        print(f"Plant {i}/{len(filtered_plants)}: {plant_id}")
        print(f"Progress: {i/len(filtered_plants)*100:.1f}%")
        print(f"{'#' * 80}")
        
        success = run_plant_experiments(plant_config_path, resume=resume, output_dir=output_dir)
        total_success += success
        total_experiments += 284
        plants_processed += 1
        
        # Estimate remaining time
        elapsed = time.time() - start_time
        avg_time_per_plant = elapsed / plants_processed
        remaining_plants = len(filtered_plants) - plants_processed
        estimated_remaining = avg_time_per_plant * remaining_plants
        
        print(f"\nProgress Summary:")
        print(f"  Plants processed: {plants_processed}/{len(filtered_plants)}")
        print(f"  Experiments done: {total_success}/{total_experiments}")
        print(f"  Time elapsed:     {elapsed/3600:.2f} hours")
        print(f"  Time remaining:   {estimated_remaining/3600:.2f} hours")
        print(f"  Est. completion:  {(elapsed + estimated_remaining)/3600:.2f} hours total")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Batch Experiments Finished!")
    print("=" * 80)
    print(f"Plants processed: {plants_processed}")
    print(f"Experiments successful: {total_success}/{total_experiments} ({total_success/total_experiments*100:.1f}%)")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg per plant: {total_time/plants_processed/60:.1f} minutes" if plants_processed > 0 else "")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run multi-plant PV forecasting experiments with resume support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all plants with resume support (default)
  python run_experiments_multi_plant.py
  
  # Run first 25 plants
  python run_experiments_multi_plant.py --max-plants 25
  
  # Run plants 26-50 (skip first 25, run next 25)
  python run_experiments_multi_plant.py --skip 25 --max-plants 25
  
  # Run specific plants
  python run_experiments_multi_plant.py --plants 1001 1002 1003
  
  # Run single plant
  python run_experiments_multi_plant.py --plant 1140
  
  # Start fresh without resume
  python run_experiments_multi_plant.py --no-resume
  
  # Scan status only
  python run_experiments_multi_plant.py --status-only
        """
    )
    
    parser.add_argument('--plant', type=str, 
                       help='Single plant ID to run (e.g., 1140)')
    parser.add_argument('--plants', nargs='+', 
                       help='List of specific plant IDs to run (e.g., 1001 1002 1003)')
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip first N plants (default: 0)')
    parser.add_argument('--max-plants', type=int, default=None,
                       help='Maximum number of plants to process (default: all)')
    parser.add_argument('--no-resume', action='store_true', 
                       help='Start fresh without resuming from previous results')
    parser.add_argument('--status-only', action='store_true',
                       help='Only show status without running experiments')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (default: current directory). '
                            'For Colab/Drive: /content/drive/MyDrive/Solar_PV_electricity/results')
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    
    # Status-only mode
    if args.status_only:
        print("=" * 80)
        print("Plant Experiments Status Scan")
        print("=" * 80)
        
        if args.output_dir:
            print(f"Checking results in: {args.output_dir}\n")
        
        statuses = scan_all_plants_status(args.output_dir)
        
        if not statuses:
            print("No plant configurations found")
            sys.exit(0)
        
        # Summary statistics
        complete = sum(1 for s in statuses if s['is_complete'])
        in_progress = sum(1 for s in statuses if 0 < s['completed_experiments'] < 284)
        not_started = sum(1 for s in statuses if s['completed_experiments'] == 0)
        total_completed_exps = sum(s['completed_experiments'] for s in statuses)
        total_exps = len(statuses) * 284
        
        print(f"\nOverall Statistics:")
        print(f"  Total plants:   {len(statuses)}")
        print(f"  [COMPLETE]:     {complete} plants")
        print(f"  [IN_PROGRESS]:  {in_progress} plants")
        print(f"  [NOT_STARTED]:  {not_started} plants")
        print(f"  Experiments:    {total_completed_exps}/{total_exps} ({total_completed_exps/total_exps*100:.1f}%)")
        
        print(f"\nDetailed Status:")
        print(f"{'Plant ID':<10} {'Status':<15} {'Done':<10} {'Remain':<10} {'Result File':<40}")
        print("-" * 90)
        
        for status in statuses:
            status_str = status['status']
            if status_str == 'COMPLETE':
                status_display = '[COMPLETE]'
            elif status_str == 'IN_PROGRESS':
                status_display = '[IN_PROGRESS]'
            else:
                status_display = '[NOT_STARTED]'
            
            result_file = os.path.basename(status['result_file']) if status['result_file'] else 'N/A'
            
            print(f"{status['plant_id']:<10} {status_display:<15} "
                  f"{status['completed_experiments']:<10} "
                  f"{status['remaining_experiments']:<10} "
                  f"{result_file:<40}")
        
        print("\n" + "=" * 80)
        sys.exit(0)
    
    # Single plant mode
    if args.plant:
        plant_config_path = f"config/plants/Plant{args.plant}.yaml"
        if not os.path.exists(plant_config_path):
            print(f"[ERROR] Plant config not found: {plant_config_path}")
            print(f"\nAvailable plants:")
            manager = PlantConfigManager()
            plants = manager.get_all_plants()
            for plant in plants[:20]:
                print(f"  - Plant {plant['plant_id']}")
            if len(plants) > 20:
                print(f"  ... and {len(plants)-20} more plants")
            sys.exit(1)
        
        run_plant_experiments(plant_config_path, resume=resume, output_dir=args.output_dir)
    else:
        # Run multiple plants
        run_all_plants(resume=resume, skip=args.skip, 
                      max_plants=args.max_plants, plants=args.plants, output_dir=args.output_dir)