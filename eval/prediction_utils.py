#!/usr/bin/env python3
"""
Prediction extraction utilities for single-step evaluation
Extract 24-hour ahead predictions for day-ahead forecasting scenario
"""

import numpy as np


def extract_one_hour_ahead_predictions(predictions, ground_truth):
    """
    For daily prediction method: predictions are already day-ahead
    Just flatten the (n_days, 24) arrays to 1D for CSV saving
    
    In daily prediction mode:
        - Each sample is ONE day's prediction (made at 23:00 previous day)
        - predictions[i] = full 24h forecast for day i+1
        - ground_truth[i] = actual 24h values for day i+1
    
    This function simply flattens for compatibility with CSV saving logic.
    
    Args:
        predictions: (n_days, 24) - each row is one day's 24h prediction
        ground_truth: (n_days, 24) - actual values
    
    Returns:
        final_predictions: (n_days * 24,) - flattened predictions
        final_ground_truth: (n_days * 24,) - flattened ground truth
    """
    # Simply flatten - each prediction is already day-ahead
    final_preds = predictions.flatten()
    final_gt = ground_truth.flatten()
    
    n_days = len(predictions)
    print(f"  Day-ahead evaluation: {n_days} daily predictions ({len(final_preds)} hourly values)")
    
    return final_preds, final_gt


def extract_multi_horizon_predictions(predictions, ground_truth, horizons=[1, 3, 6, 12, 24]):
    """
    Extract predictions at multiple horizons for detailed analysis
    
    Args:
        predictions: (n_samples, 24)
        ground_truth: (n_samples, 24)
        horizons: List of hours ahead to evaluate [1, 3, 6, 12, 24]
    
    Returns:
        Dictionary with metrics for each horizon
    """
    results = {}
    
    for h in horizons:
        if h > predictions.shape[1]:
            continue
        
        horizon_preds = []
        horizon_gt = []
        
        # For h-hour ahead prediction
        # Use sample i-h+1's predictions[h-1] to predict sample i's ground_truth[0]
        for i in range(h-1, len(predictions)):
            source_idx = i - h + 1
            pred_h_ahead = predictions[source_idx, h-1]
            actual = ground_truth[i, 0]
            
            horizon_preds.append(pred_h_ahead)
            horizon_gt.append(actual)
        
        results[f'{h}h'] = {
            'predictions': np.array(horizon_preds),
            'ground_truth': np.array(horizon_gt),
            'n_samples': len(horizon_preds)
        }
    
    return results

