#!/usr/bin/env python3
"""
Evaluation metrics calculation utilities
Calculate MAE, RMSE, NRMSE, R², MAPE, sMAPE based on paper definitions
"""

import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics
    
    Args:
        True values (n_samples, n_hours)
        Predicted values (n_samples, n_hours)
    
    Returns:
        Dictionary containing all metrics
    """
    # Flatten to 1D array
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'nrmse': np.nan,
            'r_square': np.nan,
            'r2': np.nan,
            'smape': np.nan
        }
    
    T = len(y_true_clean)
    
    # MAE
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    y_range = np.max(y_true_clean) - np.min(y_true_clean)
    if y_range != 0:
        nrmse = rmse / y_range
    else:
        nrmse = np.nan
    
    # R²
    r_square = r2_score(y_true_clean, y_pred_clean)
    
    # Removed MAPE calculation due to large and unstable values
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    # y_t - ŷ_t| / (|y_t| + |ŷ_t|)] | Formula: sMAPE = (1/n) * Σ[2 * |y_t - ŷ_t| / (|y_t| + |ŷ_t|)]
    # For solar data, only calculate non-zero values to avoid abnormal sMAPE from zeros
    
    # Only calculate non-zero values (at least one of true value or predicted value is non-zero)
    nonzero_mask = (y_true_clean > 0) | (y_pred_clean > 0)
    
    if np.any(nonzero_mask):
        y_true_nonzero = y_true_clean[nonzero_mask]
        y_pred_nonzero = y_pred_clean[nonzero_mask]
        
        # y_t| + |ŷ_t|) | Calculate denominator (|y_t| + |ŷ_t|)
        denominator = np.abs(y_true_nonzero) + np.abs(y_pred_nonzero)
        
        # Avoid division by zero
        smape_mask = denominator > 0
        
        if np.any(smape_mask):
            smape = np.mean(2 * np.abs(y_true_nonzero[smape_mask] - y_pred_nonzero[smape_mask]) / 
                           denominator[smape_mask])
        else:
            smape = np.nan
    else:
        smape = np.nan
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'nrmse': round(nrmse, 4),
        'r_square': round(r_square, 4),
        'r2': round(r_square, 4),  # Add r2 alias
        'smape': round(smape, 4)  # Keep decimal format
    }

def calculate_mse(y_true, y_pred):
    """
    Calculate MSE (for test_loss)
    
    Args:
        True values (n_samples, n_hours)
        Predicted values (n_samples, n_hours)
    
    Returns:
        MSE value
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    return round(mse, 4)


def calculate_daily_avg_metrics(y_true, y_pred):
    """
    Calculate metrics using daily average method
    
    For day-ahead forecasting, calculate RMSE for each day separately,
    then average across all days. This approach better reflects daily
    prediction performance compared to flattening all values.
    
    Args:
        y_true: True values (n_days, 24)
        y_pred: Predicted values (n_days, 24)
    
    Returns:
        Dictionary containing daily-averaged metrics
    """
    n_days = y_true.shape[0]
    
    daily_rmses = []
    daily_maes = []
    
    # Calculate metrics for each day
    for i in range(n_days):
        day_true = y_true[i]
        day_pred = y_pred[i]
        
        # Remove NaN values
        mask = ~(np.isnan(day_true) | np.isnan(day_pred))
        day_true_clean = day_true[mask]
        day_pred_clean = day_pred[mask]
        
        if len(day_true_clean) > 0:
            daily_rmse = np.sqrt(np.mean((day_true_clean - day_pred_clean) ** 2))
            daily_mae = np.mean(np.abs(day_true_clean - day_pred_clean))
            daily_rmses.append(daily_rmse)
            daily_maes.append(daily_mae)
    
    # Average across all days
    rmse_daily_avg = np.mean(daily_rmses) if len(daily_rmses) > 0 else np.nan
    mae_daily_avg = np.mean(daily_maes) if len(daily_maes) > 0 else np.nan
    
    # Calculate R² on all flattened data (R² doesn't benefit from daily averaging)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) > 0:
        r_square = r2_score(y_true_clean, y_pred_clean)
    else:
        r_square = np.nan
    
    return {
        'mae': round(mae_daily_avg, 4),
        'rmse': round(rmse_daily_avg, 4),
        'r2': round(r_square, 4),
        'r_square': round(r_square, 4),
        'n_days': len(daily_rmses)
    }