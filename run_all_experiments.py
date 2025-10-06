#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)


from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model

def generate_all_configs():
    """
    Generate all experiment configurations
    
    Experiment counts:
    
    1. DL models (LSTM, GRU, Transformer, TCN): 4 models x 2 complexities
       - PV experiments: 4 feature combos x 2 lookbacks x 2 TE = 16 per model-complexity
       - NWP experiments: 2 feature combos x 2 TE = 4 per model-complexity
       - Subtotal: 4 models x 2 complexities x (16 + 4) = 160 experiments
    
    2. ML models (RF, XGB, LGBM): 3 models x 2 complexities
       - PV experiments: 4 feature combos x 2 lookbacks x 2 TE = 16 per model-complexity
       - NWP experiments: 2 feature combos x 2 TE = 4 per model-complexity
       - Subtotal: 3 models x 2 complexities x (16 + 4) = 120 experiments
    
    3. Linear model: special case (no lookback, no complexity, only TE)
       - NWP: 2 feature combos x 2 TE = 4 experiments
       - Subtotal: 4 experiments
    
    Total: 160 + 120 + 4 = 284 experiments
    """
    
    configs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    
    dl_models = ['LSTM', 'GRU', 'Transformer', 'TCN']
    ml_models = ['RF', 'XGB', 'LGBM']  # Linear is handled separately
    complexities = ['low', 'high']
    
    lookbacks = [24, 72]
    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]
    te_options = [True, False]
    
    # DL models experiments
    for model in dl_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        config = create_config(
                            data_path, model, complexity, lookback, 
                            feat_combo, use_te, False  # is_nwp_only=False
                        )
                        configs.append(config)
    
    # ML models experiments  
    for model in ml_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        config = create_config(
                            data_path, model, complexity, lookback, 
                            feat_combo, use_te, False  # is_nwp_only=False
                        )
                        configs.append(config)
    
    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]
    
    # DL models for NWP-only experiments
    for model in dl_models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    config = create_config(
                        data_path, model, complexity, 0,  # lookback=0 for NWP
                        feat_combo, use_te, True  # is_nwp_only=True
                    )
                    configs.append(config)
    
    # ML models for NWP-only experiments
    for model in ml_models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    config = create_config(
                        data_path, model, complexity, 0,  # lookback=0 for NWP
                        feat_combo, use_te, True  # is_nwp_only=True
                    )
                    configs.append(config)
    
    # Linear model: special case (only NWP/NWP+, no lookback, no complexity, only TE)
    for feat_combo in feature_combos_nwp:
        for use_te in te_options:
            config = create_config(
                data_path, 'Linear', None, 0,  # complexity=None for Linear
                feat_combo, use_te, True  # is_nwp_only=True
            )
            configs.append(config)
    
    return configs

def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
    
    config = {
        'data_path': data_path,
        'model': model,
        'model_complexity': complexity,
        'use_pv': feat_combo['use_pv'],
        'use_hist_weather': feat_combo['use_hist_weather'],
        'use_forecast': feat_combo['use_forecast'],
        'use_ideal_nwp': feat_combo['use_ideal_nwp'],
        'use_time_encoding': use_te,
        'weather_category': 'all_weather',
        'future_hours': 24,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_summary': False,
            'save_excel_results': False,
            'save_training_log': False
        }
    }
    
    if is_nwp_only:
        config['past_hours'] = 0
        config['past_days'] = 0
        config['no_hist_power'] = True
        feat_name = feat_combo['name']
    else:
        config['past_hours'] = lookback
        config['past_days'] = lookback // 24
        config['no_hist_power'] = False
        feat_name = f"{feat_combo['name']}_{lookback}h"
    
    # Common data split ratios
    config.update({
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
    })
    
    # Model-specific parameters
    if model in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        # DL model parameters
        if complexity == 'low':
            config.update({
                'train_params': {
                    'epochs': 20,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'patience': 10,
                    'min_delta': 0.001,
                    'weight_decay': 1e-4
                },
                'model_params': {
                    'd_model': 16,
                    'hidden_dim': 8,
                    'num_heads': 2,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'tcn_channels': [8, 16],
                    'kernel_size': 3
                }
            })
        else:  # high complexity
            config.update({
                'train_params': {
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'patience': 10,
                    'min_delta': 0.001,
                    'weight_decay': 1e-4
                },
                'model_params': {
                    'd_model': 32,
                    'hidden_dim': 16,
                    'num_heads': 2,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'tcn_channels': [16, 32],
                    'kernel_size': 3
                }
            })
    elif model == 'Linear':
        # Linear model: no complexity parameter (always None)
        config['model_params'] = {}  # Linear regression has no hyperparameters
    else:
        # ML model parameters (RF, XGB, LGBM)
        if complexity == 'low':
            config['model_params'] = {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0
            }
        else:  # high complexity
            config['model_params'] = {
                'n_estimators': 200,
                'max_depth': 15,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbosity': 0
            }
    
    te_suffix = 'TE' if use_te else 'noTE'
    
    # For Linear model, don't include complexity in experiment name
    if model == 'Linear':
        config['experiment_name'] = f"{model}_{feat_name}_{te_suffix}"
    else:
        config['experiment_name'] = f"{model}_{complexity}_{feat_name}_{te_suffix}"
    
    config['save_dir'] = f'results/{config["experiment_name"]}'
    
    return config

def run_all_experiments():
    
    print("="*80)
    print("="*80)
    
    all_configs = generate_all_configs()
    
    import torch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"all_experiments_results_{timestamp}.csv"
    results_df = pd.DataFrame(columns=[
        'experiment_name', 'model', 'complexity', 'feature_combo', 
        'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2', 
        'train_time_sec', 'test_samples', 'best_epoch', 'param_count'
    ])
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
            
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, config['past_hours'], config['future_hours'], 
                hist_feats, fcst_feats, no_hist_power
            )
            
            total_samples = len(X_hist)
            indices = np.arange(total_samples)
            
            train_size = int(total_samples * config['train_ratio'])
            val_size = int(total_samples * config['val_ratio'])
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            X_hist_train, y_train = X_hist[train_idx], y[train_idx]
            X_hist_val, y_val = X_hist[val_idx], y[val_idx]
            X_hist_test, y_test = X_hist[test_idx], y[test_idx]
            
            if X_fcst is not None:
                X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
            else:
                X_fcst_train, X_fcst_val, X_fcst_test = None, None, None
            
            train_hours = np.array([hours[i] for i in train_idx])
            val_hours = np.array([hours[i] for i in val_idx])
            test_hours = np.array([hours[i] for i in test_idx])
            test_dates = [dates[i] for i in test_idx]
            
            train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
            val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
            test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            # Choose training function based on model type
            if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            elif config['model'] in ['RF', 'XGB', 'LGBM', 'Linear']:
                model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
            else:
                raise ValueError(f"Unknown model type: {config['model']}")
            
            training_time = time.time() - start_time
            
            parts = exp_name.split('_')
            if len(parts) >= 3:
                feat_parts = parts[2:-1]
                feat_name_str = '_'.join(feat_parts)
            else:
                feat_name_str = 'unknown'
            
            result = {
                'experiment_name': exp_name,
                'model': config['model'],
                'complexity': config['model_complexity'],
                'feature_combo': feat_name_str,
                'lookback_hours': config['past_hours'],
                'use_time_encoding': config['use_time_encoding'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'train_time_sec': training_time,
                'test_samples': metrics['samples_count'],
                'best_epoch': metrics.get('best_epoch', 0),
                'param_count': metrics.get('param_count', 0)
            }
            
            results.append(result)
            
            print(f"  [OK] MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            result = {
                'experiment_name': exp_name,
                'model': config['model'],
                'complexity': config['model_complexity'],
                'feature_combo': 'FAILED',
                'lookback_hours': config['past_hours'],
                'use_time_encoding': config['use_time_encoding'],
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'train_time_sec': 0,
                'test_samples': 0,
                'best_epoch': 0,
                'param_count': 0
            }
            results.append(result)
            
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            continue
    
    if results:
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        
        final_df = pd.read_csv(output_file)
        
        
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        top_10 = final_df.nsmallest(10, 'rmse')[['experiment_name', 'rmse', 'mae']]
        print(top_10.to_string(index=False))
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = run_all_experiments()
    if not success:
        sys.exit(1)
