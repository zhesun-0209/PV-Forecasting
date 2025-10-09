"""
eval/eval_utils.py

Utilities to save summary, predictions, training logs, and call plotting routines.
"""

import os
import pandas as pd
import numpy as np
from eval.excel_utils import save_plant_excel_results

# ===== Define Deep Learning model names =====
DL_MODELS = {"Transformer", "LSTM", "GRU", "TCN"}

def save_results(
    model,
    metrics: dict,
    dates: list,
    y_true: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    config: dict
):
    """
    Save summary.csv, predictions.csv, training_log.csv, and generate plots
    under config['save_dir'].

    Args:
        model:   Trained DL or sklearn model
        metrics: Dictionary containing:
                 'test_loss', 'train_time_sec', 'param_count', 'rmse', 'mae',
                 'predictions' (n,h), 'y_true' (n,h),
                 'dates' (n), 'epoch_logs' (list of dicts)
        dates:   List of datetime strings
        y_true, Xh_test, Xf_test: Used for legacy or optional plots
        config:  Dictionary with keys like 'save_dir', 'model', 'scaler_target'
    """
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Extract predictions and ground truth (already computed in metrics)
    preds = metrics['predictions']
    yts   = metrics['y_true']
    
    # Use pre-computed metrics instead of recalculating
    test_mse = metrics['mse']
    test_rmse = metrics['rmse']
    test_mae = metrics['mae']
    r_square = metrics['r_square']
    
    save_options = config.get('save_options', {})
    
    summary = {
        'model':           config['model'],
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast':    config.get('use_forecast', False),
        'past_days':       config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'correlation_level': config.get('correlation_level', 'high'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_hours':      config['past_hours'],
        'future_hours':    config['future_hours'],
        
        'mse':             test_mse,
        'rmse':            test_rmse,
        'mae':             test_mae,
        'r_square':        r_square,
        
        'train_time_sec':  metrics.get('train_time_sec'),
        'inference_time_sec': metrics.get('inference_time_sec', np.nan),
        'param_count':     metrics.get('param_count'),
        'samples_count':   len(preds),
    }

    # ===== 2. Save predictions.csv =====
    if save_options.get('save_predictions', True):
        hrs = metrics.get('hours')
        dates_list = metrics.get('dates', dates)
        records = []
        n_samples, horizon = preds.shape
        
        # Handle case where hours information is not available
        if hrs is None:
            # Generate default hour sequence if not provided
            hrs = np.tile(np.arange(horizon), (n_samples, 1))
        
        for i in range(n_samples):
            start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
            for h in range(horizon):
                dt = start + pd.Timedelta(hours=h)
                records.append({
                    'window_index':      i,
                    'forecast_datetime': dt,
                    'hour':              int(hrs[i, h]) if hrs is not None else dt.hour,
                    'y_true':            float(yts[i, h]),
                    'y_pred':            float(preds[i, h])
                })
        pd.DataFrame(records).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # ===== 3. Save training log (only if DL) =====
    is_dl = config['model'] in DL_MODELS
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_training_log', True):
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )

    # ===== 4. Save Excel results =====
    if save_options.get('save_excel_results', True):
        result_data = {
            'config': {
                'model': config['model'],
                'use_pv': config.get('use_pv', True),
                'use_hist_weather': config.get('use_hist_weather', False),
                'use_forecast': config.get('use_forecast', False),
                'weather_category': config.get('weather_category', 'irradiance'),
                'use_time_encoding': config.get('use_time_encoding', True),
                'past_days': config.get('past_days', 1),
                'model_complexity': config.get('model_complexity', 'low'),
                'epochs': config.get('epochs', 15),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 0.001)
            },
            'metrics': {
                'train_time_sec': summary['train_time_sec'],
                'inference_time_sec': summary['inference_time_sec'],
                'param_count': summary['param_count'],
                'samples_count': summary['samples_count'],
                'mse': summary['mse'],
                'rmse': summary['rmse'],
                'mae': summary['mae'],
                'nrmse': metrics.get('nrmse', np.nan),
                'r_square': summary['r_square'],
                'smape': metrics.get('smape', np.nan),
                'best_epoch': metrics.get('best_epoch', np.nan),
                'final_lr': metrics.get('final_lr', np.nan),
                'gpu_memory_used': metrics.get('gpu_memory_used', 0)
            }
        }
        
        from eval.excel_utils import append_plant_excel_results
        csv_file = append_plant_excel_results(
            plant_id=config.get('plant_id', 'unknown'),
            result=result_data,
            save_dir=save_dir
        )
    else:
        print(f"[INFO] Results saved in {save_dir}")

def save_season_hour_results(
    model,
    metrics: dict,
    dates: list,
    y_true: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    config: dict
):
    """
    season and hour analysis
    prediction.csv
    
    Args:
        Xh_test, Xf_test:
    """
    drive_path = "/content/drive/MyDrive/Solar PV electricity/hour and season analysis"
    os.makedirs(drive_path, exist_ok=True)
    
    preds = metrics['predictions']
    yts = metrics['y_true']
    
    test_mse = np.mean((preds - yts) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(preds - yts))
    
    y_mean = np.mean(yts)
    ss_tot = np.sum((yts - y_mean) ** 2)
    ss_res = np.sum((yts - preds) ** 2)
    r_square = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    nrmse = (test_rmse / (test_mae + 1e-8)) * 100 if test_mae > 0 else 0
    smape = (2 * test_mae / (test_mae + 1e-8)) * 100 if test_mae > 0 else 0
    
    project_id = config.get('plant_id', 'unknown')
    model_name = config.get('model', 'unknown')
    
    prediction_file = os.path.join(drive_path, f"{project_id}_prediction.csv")
    
    hrs = metrics.get('hours')
    dates_list = metrics.get('dates', dates)
    records = []
    n_samples, horizon = preds.shape
    
    if hrs is None:
        hrs = np.tile(np.arange(horizon), (n_samples, 1))
    
    for i in range(n_samples):
        start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
        for h in range(horizon):
            dt = start + pd.Timedelta(hours=h)
            records.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'ground_truth': float(yts[i, h]),
                'prediction': float(preds[i, h]),
                'model': model_name,
                'project_id': project_id,
                'window_index': i,
                'hour': int(hrs[i, h]) if hrs is not None else dt.hour
            })
    
    pred_df = pd.DataFrame(records)
    if os.path.exists(prediction_file):
        existing_df = pd.read_csv(prediction_file)
        pred_df = pd.concat([existing_df, pred_df], ignore_index=True)
    pred_df.to_csv(prediction_file, index=False)
    
    summary_file = os.path.join(drive_path, f"{project_id}_summary.csv")
    
    summary_data = {
        'model': model_name,
        'weather_level': config.get('weather_category', 'unknown'),
        'lookback_hours': config.get('past_hours', 24),
        'complexity_level': config.get('model_complexity', 'unknown').replace('level', ''),
        'dataset_scale': '80%',
        'use_pv': config.get('use_pv', False),
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'use_time_encoding': config.get('use_time_encoding', False),
        'past_days': config.get('past_days', 1),
        'use_ideal_nwp': config.get('use_ideal_nwp', False),
        'selected_weather_features': str(config.get('selected_weather_features', [])),
        'epochs': config.get('epochs', 0),
        'batch_size': config.get('train_params', {}).get('batch_size', 0),
        'learning_rate': config.get('train_params', {}).get('learning_rate', 0.0),
        'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
        'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
        'param_count': metrics.get('param_count', 0),
        'samples_count': len(preds),
        'best_epoch': metrics.get('best_epoch', 0),
        'final_lr': metrics.get('final_lr', 0.0),
        'mse': round(test_mse, 4),
        'rmse': round(test_rmse, 4),
        'mae': round(test_mae, 4),
        'nrmse': round(nrmse, 4),
        'r_square': round(r_square, 4),
        'smape': round(smape, 4),
        'gpu_memory_used': metrics.get('gpu_memory_used', 0),
        'config_file': f"season_hour_{model_name.lower()}.yaml"
    }
    
    summary_df = pd.DataFrame([summary_data])
    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_file, index=False)
    
    
    return summary_data
