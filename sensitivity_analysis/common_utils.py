#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for sensitivity analysis experiments
Shared functions for running experiments and computing metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import PlantConfigManager
from data.data_utils import load_raw_data, preprocess_features, create_daily_windows, split_data
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model


# Model definitions
DL_MODELS = ['LSTM', 'GRU', 'Transformer', 'TCN']
ML_MODELS = ['RF', 'XGB', 'LGBM']
ALL_MODELS_NO_LINEAR = DL_MODELS + ML_MODELS  # 7 models excluding Linear


def get_season(month: int) -> str:
    """
    Get season from month
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name: Spring, Summer, Fall, Winter
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:  # 12, 1, 2
        return 'Winter'


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized RMSE (RMSE / mean of true values)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        NRMSE value
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mean_true = np.mean(y_true)
    if mean_true == 0:
        return np.inf
    return rmse / mean_true


def create_base_config(plant_config: Dict, model: str, complexity: str = 'high', 
                       lookback: int = 24, use_te: bool = False) -> Dict:
    """
    Create base experiment configuration
    
    Args:
        plant_config: Plant configuration
        model: Model name
        complexity: Model complexity ('low', 'mid_low', 'mid_high', 'high')
        lookback: Lookback window hours
        use_te: Use time encoding
        
    Returns:
        Experiment configuration dictionary
    """
    config = {
        'plant_id': plant_config['plant_id'],
        'data_path': plant_config['data_path'],
        'model': model,
        'model_complexity': complexity,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': use_te,
        'past_hours': lookback,
        'past_days': lookback // 24,
        'future_hours': plant_config.get('future_hours', 24),
        'train_ratio': plant_config.get('train_ratio', 0.8),
        'val_ratio': plant_config.get('val_ratio', 0.1),
        'test_ratio': plant_config.get('test_ratio', 0.1),
        'shuffle_split': True,
        'random_seed': plant_config.get('random_seed', 42),
        'weather_category': plant_config.get('weather_category', 'all_weather'),
        'start_date': plant_config.get('start_date', '2022-01-01'),
        'end_date': plant_config.get('end_date', '2024-09-28'),
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False
        }
    }
    
    # Add model parameters
    if model in DL_MODELS:
        # Get DL parameters from plant config
        dl_params = plant_config.get('dl_params', {}).get(complexity, {})
        config['train_params'] = {
            'epochs': dl_params.get('epochs', 50),
            'batch_size': dl_params.get('batch_size', 64),
            'learning_rate': dl_params.get('learning_rate', 0.001),
            'patience': dl_params.get('patience', 10),
            'min_delta': dl_params.get('min_delta', 0.001),
            'weight_decay': dl_params.get('weight_decay', 0.0001)
        }
        config['model_params'] = {
            'd_model': dl_params.get('d_model', 32),
            'hidden_dim': dl_params.get('hidden_dim', 16),
            'num_heads': dl_params.get('num_heads', 2),
            'num_layers': dl_params.get('num_layers', 2),
            'dropout': dl_params.get('dropout', 0.1),
        }
    elif model in ML_MODELS:
        # Get ML parameters from plant config
        ml_params = plant_config.get('ml_params', {}).get(complexity, {})
        config['model_params'] = ml_params
    elif model == 'Linear':
        config['model_params'] = {}
    
    return config


def run_single_experiment(config: Dict, df: pd.DataFrame) -> Dict:
    """
    Run a single experiment
    
    Args:
        config: Experiment configuration
        df: Raw dataframe with Datetime column
        
    Returns:
        Result dictionary with metrics
    """
    import time
    
    try:
        start_time = time.time()
        
        # Preprocess features
        df_processed = preprocess_features(df, config)
        
        # Train and evaluate
        model_name = config['model']
        if model_name in DL_MODELS:
            result = train_dl_model(config, df_processed)
        else:  # ML models and Linear
            result = train_ml_model(config, df_processed)
        
        train_time = time.time() - start_time
        
        # Return result
        return {
            'model': model_name,
            'complexity': config.get('model_complexity', 'N/A'),
            'mae': result['test_mae'],
            'rmse': result['test_rmse'],
            'r2': result['test_r2'],
            'train_time': train_time,
            'test_samples': result.get('test_samples', 0),
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"Error in experiment: {str(e)}")
        return {
            'model': config.get('model', 'Unknown'),
            'complexity': config.get('model_complexity', 'N/A'),
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'train_time': 0,
            'test_samples': 0,
            'status': 'FAILED',
            'error': str(e)
        }


def aggregate_results(results: List[Dict], group_by: str = None) -> pd.DataFrame:
    """
    Aggregate results across multiple plants
    
    Args:
        results: List of result dictionaries
        group_by: Column to group by (e.g., 'model', 'season', 'hour')
        
    Returns:
        DataFrame with mean and std for each group
    """
    df = pd.DataFrame(results)
    
    # Remove failed experiments
    df = df[df['status'] == 'SUCCESS']
    
    if len(df) == 0:
        print("Warning: No successful experiments to aggregate")
        return pd.DataFrame()
    
    # Compute NRMSE
    if 'rmse' in df.columns and 'mean_true' in df.columns:
        df['nrmse'] = df['rmse'] / df['mean_true']
    
    # Metrics to aggregate
    metrics = ['mae', 'rmse', 'r2', 'nrmse', 'train_time']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if group_by and group_by in df.columns:
        # Group by specified column
        grouped = df.groupby(group_by)[available_metrics]
        
        # Compute mean and std
        mean_df = grouped.mean().round(2)
        std_df = grouped.std().round(2)
        
        # Combine mean and std
        result_df = pd.DataFrame()
        for metric in available_metrics:
            result_df[f'{metric}_mean'] = mean_df[metric]
            result_df[f'{metric}_std'] = std_df[metric]
        
        return result_df
    else:
        # Overall statistics
        mean_series = df[available_metrics].mean().round(2)
        std_series = df[available_metrics].std().round(2)
        
        result_df = pd.DataFrame({
            'mean': mean_series,
            'std': std_series
        })
        
        return result_df


def load_all_plant_configs(data_dir: str = 'data') -> List[Dict]:
    """
    Load all plant configurations from CSV files in data directory
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of plant configuration dictionaries
    """
    plant_configs = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found")
        return plant_configs
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    for csv_file in csv_files:
        # Extract plant ID from filename (e.g., Project1140.csv -> 1140)
        plant_id = csv_file.replace('Project', '').replace('.csv', '')
        
        # Create minimal plant config
        plant_config = {
            'plant_id': plant_id,
            'data_path': os.path.join(data_dir, csv_file),
            'future_hours': 24,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'random_seed': 42,
            'weather_category': 'all_weather',
            'start_date': '2022-01-01',
            'end_date': '2024-09-28',
            # DL parameters - high complexity
            'dl_params': {
                'high': {
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'patience': 10,
                    'min_delta': 0.001,
                    'weight_decay': 0.0001,
                    'd_model': 32,
                    'hidden_dim': 16,
                    'num_heads': 2,
                    'num_layers': 2,
                    'dropout': 0.1
                }
            },
            # ML parameters - high complexity
            'ml_params': {
                'high': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            }
        }
        
        plant_configs.append(plant_config)
    
    return plant_configs


def save_results(results_df: pd.DataFrame, output_file: str):
    """
    Save results to CSV file
    
    Args:
        results_df: Results DataFrame
        output_file: Output CSV file path
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(output_file, index=True, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")

