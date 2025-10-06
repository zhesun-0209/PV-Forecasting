#!/usr/bin/env python3
"""
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def save_plant_excel_results(
    plant_id: str,
    results: List[Dict[str, Any]],
    save_dir: str
):
    """
    
    Args:
    """
    
    save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(save_dir, exist_ok=True)
    
    excel_data = []
    
    for result in results:
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        
        row_data = {
            'model': config.get('model', ''),
            'use_pv': config.get('use_pv', True),
            'use_hist_weather': config.get('use_hist_weather', False),
            'use_forecast': config.get('use_forecast', False),
            'weather_category': config.get('weather_category', 'irradiance'),
            'use_time_encoding': config.get('use_time_encoding', True),
            'past_days': config.get('past_days', 1),
            'model_complexity': config.get('model_complexity', 'low'),
            'epochs': config.get('epochs', 15),
            'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'use_ideal_nwp': config.get('use_ideal_nwp', False),
            
            'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
            'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
            'param_count': metrics.get('param_count', 0),
            'samples_count': metrics.get('samples_count', 0),
            'best_epoch': metrics.get('best_epoch', np.nan),
            'final_lr': metrics.get('final_lr', np.nan),
            
            'mse': round(metrics.get('mse', 0), 4),
            'rmse': round(metrics.get('rmse', 0), 4),
            'mae': round(metrics.get('mae', 0), 4),
            'nrmse': round(metrics.get('nrmse', 0), 4),
            'r_square': round(metrics.get('r_square', 0), 4),
            'smape': round(metrics.get('smape', 0), 4),
            'gpu_memory_used': metrics.get('gpu_memory_used', 0)
        }
        
        excel_data.append(row_data)
    
    df = pd.DataFrame(excel_data)
    
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    df.to_excel(excel_path, index=False)
    

def load_plant_excel_results(plant_id: str, save_dir: str) -> pd.DataFrame:
    """
    
    Args:
        
    Returns:
        DataFrame:
    """
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    if not os.path.exists(excel_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        return pd.DataFrame()

def append_plant_excel_results(
    plant_id: str,
    result: Dict[str, Any],
    save_dir: str
):
    """
    
    Args:
    """
    
    save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(save_dir, exist_ok=True)
    
    config = result.get('config', {})
    metrics = result.get('metrics', {})
    
    row_data = {
        'model': config.get('model', ''),
        'use_pv': config.get('use_pv', True),
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'weather_category': config.get('weather_category', 'irradiance'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_days': config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'epochs': config.get('epochs', 15),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'use_ideal_nwp': config.get('use_ideal_nwp', False),
        
        'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
        'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
        'param_count': metrics.get('param_count', 0),
        'samples_count': metrics.get('samples_count', 0),
        'best_epoch': metrics.get('best_epoch', np.nan),
        'final_lr': metrics.get('final_lr', np.nan),
        
        'mse': round(metrics.get('mse', 0), 4),
        'rmse': round(metrics.get('rmse', 0), 4),
        'mae': round(metrics.get('mae', 0), 4),
        'nrmse': round(metrics.get('nrmse', 0), 4),
        'r_square': round(metrics.get('r_square', 0), 4),
        'smape': round(metrics.get('smape', 0), 4),
        'gpu_memory_used': metrics.get('gpu_memory_used', 0)
    }
    
    csv_path = os.path.join(save_dir, f"{plant_id}_results.csv")
    
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            
            key_columns = ['model', 'use_pv', 'use_hist_weather', 'use_forecast', 
                          'weather_category', 'use_time_encoding', 'past_days', 'model_complexity']
            
            new_row_df = pd.DataFrame([row_data])
            
            is_duplicate = False
            for _, existing_row in existing_df.iterrows():
                if all(existing_row[col] == row_data[col] for col in key_columns):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                return csv_path
            
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
        except Exception as e:
            combined_df = pd.DataFrame([row_data])
    else:
        combined_df = pd.DataFrame([row_data])
    
    combined_df.to_csv(csv_path, index=False)
    
    
    return csv_path

def get_existing_experiments(plant_id: str, save_dir: str) -> set:
    """
    
    Args:
        
    Returns:
    """
    df = load_plant_excel_results(plant_id, save_dir)
    
    if df.empty:
        return set()
    
    existing_experiments = set()
    
    for _, row in df.iterrows():
        model = row['model']
        use_pv = row['use_pv']
        use_hist_weather = row['use_hist_weather']
        use_forecast = row['use_forecast']
        weather_category = row['weather_category']
        use_time_encoding = row['use_time_encoding']
        past_days = row['past_days']
        model_complexity = row['model_complexity']
        
        time_str = "time" if use_time_encoding else "notime"
        weather_str = weather_category if weather_category != 'none' else 'none'
        
        if past_days == 0:
            if model == 'Linear':
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist"
            else:
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist_comp{model_complexity}"
        else:
            if model == 'Linear':
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}"
            else:
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}_comp{model_complexity}"
        
        exp_id = f"{model}_{feat_str}"
        existing_experiments.add(exp_id)
    
    return existing_experiments