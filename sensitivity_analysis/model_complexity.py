#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 5: Model Complexity

Analyze model performance with different complexity levels
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) - Linear NOT included
- Configuration: PV+NWP, 24-hour lookback, no TE
- Complexity levels: 
  * low: Small models (epochs=20, d_model=16, hidden=8, layers=1, n_estimators=50)
  * mid_low: Medium-small models (epochs=35, d_model=24, hidden=12, layers=1, n_estimators=100)
  * mid_high: Medium-large models (epochs=50, d_model=32, hidden=16, layers=2, n_estimators=150)
  * high: Large models (epochs=50, d_model=32, hidden=16, layers=2, n_estimators=200)
- Metrics: MAE, RMSE, R2, NRMSE, train_time (mean and std across 100 plants)
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity_analysis.common_utils import (
    DL_MODELS, ML_MODELS, ALL_MODELS_NO_LINEAR,
    compute_nrmse,
    load_all_plant_configs
)
from data.data_utils import load_raw_data, preprocess_features


# Complexity levels
COMPLEXITY_LEVELS = ['low', 'mid_low', 'mid_high', 'high']


def create_complexity_config(plant_config, model, complexity, lookback=24, use_te=False):
    """Create configuration with specific complexity level"""
    
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
    
    # Define complexity parameters
    if model in DL_MODELS:
        if complexity == 'low':
            config['train_params'] = {
                'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001,
                'patience': 10, 'min_delta': 0.001, 'weight_decay': 0.0001
            }
            config['model_params'] = {
                'd_model': 16, 'hidden_dim': 8, 'num_heads': 2, 'num_layers': 1, 'dropout': 0.1
            }
        elif complexity == 'mid_low':
            config['train_params'] = {
                'epochs': 35, 'batch_size': 64, 'learning_rate': 0.001,
                'patience': 10, 'min_delta': 0.001, 'weight_decay': 0.0001
            }
            config['model_params'] = {
                'd_model': 24, 'hidden_dim': 12, 'num_heads': 2, 'num_layers': 1, 'dropout': 0.1
            }
        elif complexity == 'mid_high':
            config['train_params'] = {
                'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001,
                'patience': 10, 'min_delta': 0.001, 'weight_decay': 0.0001
            }
            config['model_params'] = {
                'd_model': 32, 'hidden_dim': 16, 'num_heads': 2, 'num_layers': 2, 'dropout': 0.1
            }
        else:  # high
            config['train_params'] = {
                'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001,
                'patience': 10, 'min_delta': 0.001, 'weight_decay': 0.0001
            }
            config['model_params'] = {
                'd_model': 32, 'hidden_dim': 16, 'num_heads': 2, 'num_layers': 2, 'dropout': 0.1
            }
            
    elif model in ML_MODELS:
        if complexity == 'low':
            config['model_params'] = {
                'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1,
                'min_samples_split': 10, 'min_samples_leaf': 5
            }
        elif complexity == 'mid_low':
            config['model_params'] = {
                'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.1,
                'min_samples_split': 7, 'min_samples_leaf': 3
            }
        elif complexity == 'mid_high':
            config['model_params'] = {
                'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.1,
                'min_samples_split': 5, 'min_samples_leaf': 2
            }
        else:  # high
            config['model_params'] = {
                'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
                'min_samples_split': 5, 'min_samples_leaf': 2
            }
    
    return config


def run_model_complexity_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results'):
    """
    Run model complexity analysis across all plants
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 5: Model Complexity")
    print("=" * 80)
    
    # Load all plant configurations
    plant_configs = load_all_plant_configs(data_dir)
    print(f"\nLoaded {len(plant_configs)} plant configurations")
    
    if len(plant_configs) == 0:
        print("Error: No plant configurations found")
        return
    
    # Models to test (7 models, no Linear)
    models_to_test = ALL_MODELS_NO_LINEAR
    
    # Store results for each plant
    all_results = []
    
    # Run experiments for each plant
    for plant_idx, plant_config in enumerate(plant_configs, 1):
        plant_id = plant_config['plant_id']
        data_path = plant_config['data_path']
        
        print(f"\n{'=' * 80}")
        print(f"Plant {plant_idx}/{len(plant_configs)}: {plant_id}")
        print(f"{'=' * 80}")
        
        # Load data
        try:
            df = load_raw_data(data_path)
            df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Run experiments for each model and complexity
        for model in tqdm(models_to_test, desc=f"Plant {plant_id}"):
            for complexity in COMPLEXITY_LEVELS:
                # Create configuration
                config = create_complexity_config(plant_config, model, complexity, 
                                                lookback=24, use_te=False)
                
                try:
                    # Train and evaluate
                    from train.train_dl import train_dl_model
                    from train.train_ml import train_ml_model
                    
                    # Preprocess data
                    df_processed = preprocess_features(df.copy(), config)
                    
                    if model in DL_MODELS:
                        result = train_dl_model(config, df_processed)
                    else:
                        result = train_ml_model(config, df_processed)
                    
                    # Compute NRMSE
                    y_test_pred = result.get('y_test_pred', None)
                    if y_test_pred is not None:
                        # Get test data
                        from data.data_utils import create_daily_windows, split_data
                        
                        hist_feats = []
                        if config.get('use_pv', False):
                            hist_feats.append('Capacity_Factor_hist')
                        if config.get('use_time_encoding', False):
                            hist_feats += ['month_cos', 'month_sin', 'hour_cos', 'hour_sin']
                        
                        fcst_feats = []
                        if config.get('use_forecast', False):
                            from data.data_utils import get_weather_features_by_category
                            base_weather = get_weather_features_by_category(config['weather_category'])
                            fcst_feats = [f + '_pred' for f in base_weather]
                            if config.get('use_time_encoding', False):
                                fcst_feats += ['month_cos', 'month_sin', 'hour_cos', 'hour_sin']
                        
                        X_hist, X_fcst, y, hours, dates = create_daily_windows(
                            df_processed, 
                            config['future_hours'],
                            hist_feats,
                            fcst_feats,
                            no_hist_power=config.get('no_hist_power', False),
                            past_hours=config.get('past_hours', 24)
                        )
                        
                        _, _, _, _, _, _, _, _, y_test, _, _, _ = split_data(
                            X_hist, X_fcst, y, hours, dates,
                            train_ratio=config['train_ratio'],
                            val_ratio=config['val_ratio'],
                            shuffle=config.get('shuffle_split', True),
                            random_state=config['random_seed']
                        )
                        
                        y_true_flat = y_test.flatten()
                        y_pred_flat = y_test_pred.flatten()
                        nrmse = compute_nrmse(y_true_flat, y_pred_flat)
                    else:
                        nrmse = np.nan
                    
                    # Store result
                    all_results.append({
                        'plant_id': plant_id,
                        'model': model,
                        'complexity': complexity,
                        'mae': result['test_mae'],
                        'rmse': result['test_rmse'],
                        'r2': result['test_r2'],
                        'nrmse': nrmse,
                        'train_time': result.get('train_time', 0),
                        'samples': result.get('test_samples', 0)
                    })
                    
                except Exception as e:
                    print(f"  Error running {model} - {complexity}: {e}")
                    continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nError: No results generated")
        return
    
    print(f"\nTotal results: {len(results_df)}")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by complexity and model
    grouped = results_df.groupby(['complexity', 'model'])
    
    # Compute mean and std
    agg_results = []
    for (complexity, model), group in grouped:
        agg_results.append({
            'complexity': complexity,
            'model': model,
            'mae_mean': group['mae'].mean(),
            'mae_std': group['mae'].std(),
            'rmse_mean': group['rmse'].mean(),
            'rmse_std': group['rmse'].std(),
            'r2_mean': group['r2'].mean(),
            'r2_std': group['r2'].std(),
            'nrmse_mean': group['nrmse'].mean(),
            'nrmse_std': group['nrmse'].std(),
            'train_time_mean': group['train_time'].mean(),
            'train_time_std': group['train_time'].std(),
            'n_plants': len(group)
        })
    
    agg_df = pd.DataFrame(agg_results)
    
    # Round to 2 decimals
    for col in agg_df.columns:
        if col not in ['complexity', 'model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Order complexity levels
    agg_df['complexity'] = pd.Categorical(
        agg_df['complexity'], 
        categories=COMPLEXITY_LEVELS, 
        ordered=True
    )
    agg_df = agg_df.sort_values(['complexity', 'model'])
    
    # Pivot table
    pivot_df = agg_df.pivot(index='complexity', columns='model')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_detailed = os.path.join(output_dir, 'model_complexity_detailed.csv')
    results_df.to_csv(output_file_detailed, index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved to: {output_file_detailed}")
    
    output_file_agg = os.path.join(output_dir, 'model_complexity_aggregated.csv')
    agg_df.to_csv(output_file_agg, index=False, encoding='utf-8-sig')
    print(f"Aggregated results saved to: {output_file_agg}")
    
    output_file_pivot = os.path.join(output_dir, 'model_complexity_pivot.csv')
    pivot_df.to_csv(output_file_pivot, encoding='utf-8-sig')
    print(f"Pivot table saved to: {output_file_pivot}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (MAE by complexity and model):")
    print("=" * 80)
    summary = agg_df.pivot(index='complexity', columns='model', values='mae_mean')
    print(summary)
    
    print("\n" + "=" * 80)
    print("Model Complexity Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: Model Complexity')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_model_complexity_analysis(data_dir=args.data_dir, output_dir=args.output_dir)

