#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 4: Lookback Window Length

Analyze model performance with different lookback window lengths
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) - Linear NOT included
- Configuration: PV+NWP, various lookback windows, no TE, high complexity
- Lookback windows: 24h, 72h, 120h, 168h
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
    create_base_config,
    load_all_plant_configs
)
from data.data_utils import load_raw_data, preprocess_features, create_daily_windows, split_data


# Lookback windows to test
LOOKBACK_WINDOWS = [24, 72, 120, 168]


def run_lookback_window_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results'):
    """
    Run lookback window analysis across all plants
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 4: Lookback Window Length")
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
        
        # Run experiments for each model and lookback window
        for model in tqdm(models_to_test, desc=f"Plant {plant_id}"):
            for lookback in LOOKBACK_WINDOWS:
                # Create configuration: PV+NWP, lookback hours, no TE, high complexity
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=lookback, use_te=False)
                
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
                            past_hours=lookback
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
                        'lookback_hours': lookback,
                        'mae': result['test_mae'],
                        'rmse': result['test_rmse'],
                        'r2': result['test_r2'],
                        'nrmse': nrmse,
                        'train_time': result.get('train_time', 0),
                        'samples': result.get('test_samples', 0)
                    })
                    
                except Exception as e:
                    print(f"  Error running {model} - lookback {lookback}h: {e}")
                    continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nError: No results generated")
        return
    
    print(f"\nTotal results: {len(results_df)}")
    
    # Aggregate results by lookback window and model
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by lookback_hours and model
    grouped = results_df.groupby(['lookback_hours', 'model'])
    
    # Compute mean and std
    agg_results = []
    for (lookback, model), group in grouped:
        agg_results.append({
            'lookback_hours': lookback,
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
        if col not in ['lookback_hours', 'model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Sort by lookback hours
    agg_df = agg_df.sort_values(['lookback_hours', 'model'])
    
    # Pivot table for better visualization
    pivot_df = agg_df.pivot(index='lookback_hours', columns='model')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file_detailed = os.path.join(output_dir, 'lookback_window_detailed.csv')
    results_df.to_csv(output_file_detailed, index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved to: {output_file_detailed}")
    
    # Save aggregated results
    output_file_agg = os.path.join(output_dir, 'lookback_window_aggregated.csv')
    agg_df.to_csv(output_file_agg, index=False, encoding='utf-8-sig')
    print(f"Aggregated results saved to: {output_file_agg}")
    
    # Save pivot table
    output_file_pivot = os.path.join(output_dir, 'lookback_window_pivot.csv')
    pivot_df.to_csv(output_file_pivot, encoding='utf-8-sig')
    print(f"Pivot table saved to: {output_file_pivot}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (MAE by lookback window and model):")
    print("=" * 80)
    summary = agg_df.pivot(index='lookback_hours', columns='model', values='mae_mean')
    print(summary)
    
    print("\n" + "=" * 80)
    print("Lookback Window Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: Lookback Window Length')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_lookback_window_analysis(data_dir=args.data_dir, output_dir=args.output_dir)

