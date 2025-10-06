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
    
    y_mean = np.mean(y_true_clean)
    if y_mean != 0:
        nrmse = rmse / y_mean
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
