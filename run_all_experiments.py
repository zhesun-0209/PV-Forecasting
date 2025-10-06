#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行160次实验并保存性能指标到CSV
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

# 确保工作目录正确
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

print(f"工作目录: {os.getcwd()}")

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model

def generate_all_configs():
    """生成所有160个实验配置"""
    
    configs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    
    models = ['LSTM', 'GRU', 'Transformer', 'TCN']
    complexities = ['low', 'high']
    
    # === PV相关实验 (128次) ===
    lookbacks = [24, 72]
    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]
    te_options = [True, False]
    
    for model in models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        config = create_config(
                            data_path, model, complexity, lookback, 
                            feat_combo, use_te, False  # is_nwp_only=False
                        )
                        configs.append(config)
    
    # === NWP相关实验 (32次) ===
    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]
    
    for model in models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    config = create_config(
                        data_path, model, complexity, 0,  # lookback=0 for NWP
                        feat_combo, use_te, True  # is_nwp_only=True
                    )
                    configs.append(config)
    
    print(f"总配置数: {len(configs)}")
    return configs

def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
    """创建单个实验配置"""
    
    # 基础配置
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
    
    # 设置lookback和no_hist_power
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
    
    # 根据复杂度设置参数
    if complexity == 'low':
        config.update({
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'train_params': {
                'epochs': 20,
                'batch_size': 64,
                'learning_rate': 0.001,
                'patience': 10,
                'min_delta': 0.001,
                'weight_decay': 1e-4  # L2正则化
            },
            'model_params': {
                'd_model': 16,  # 降低复杂度
                'hidden_dim': 8,   # 降低复杂度
                'num_heads': 2,
                'num_layers': 1,   # 降低为1层
                'dropout': 0.1,
                'tcn_channels': [8, 16],  # 降低复杂度
                'kernel_size': 3
            }
        })
    else:  # high (使用原来的low配置)
        config.update({
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'train_params': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001,
                'patience': 10,
                'min_delta': 0.001,
                'weight_decay': 1e-4  # L2正则化
            },
            'model_params': {
                'd_model': 32,  # 原low配置
                'hidden_dim': 16,  # 原low配置
                'num_heads': 2,
                'num_layers': 2,  # 原low配置
                'dropout': 0.1,
                'tcn_channels': [16, 32],  # 原low配置
                'kernel_size': 3
            }
        })
    
    te_suffix = 'TE' if use_te else 'noTE'
    config['experiment_name'] = f"{model}_{complexity}_{feat_name}_{te_suffix}"
    config['save_dir'] = f'results/{config["experiment_name"]}'
    
    return config

def run_all_experiments():
    """运行所有160次实验"""
    
    print("="*80)
    print("开始运行160次深度学习模型实验")
    print("="*80)
    
    # 生成所有配置
    all_configs = generate_all_configs()
    
    # 加载数据（只加载一次）
    print("\n加载数据...")
    import torch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    print(f"数据加载完成，共{len(df)}行")
    
    # 检查GPU
    print(f"\nGPU状态: CUDA可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 存储结果
    results = []
    
    # 立即创建CSV文件（包含表头）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"all_experiments_results_{timestamp}.csv"
    results_df = pd.DataFrame(columns=[
        'experiment_name', 'model', 'complexity', 'feature_combo', 
        'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2', 
        'train_time_sec', 'test_samples', 'best_epoch', 'param_count'
    ])
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"CSV文件已创建: {output_file}\n")
    
    # 运行所有实验
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        print(f"\n{'='*80}")
        print(f"实验 {idx}/{len(all_configs)}: {exp_name}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            
            # 数据预处理
            print("  数据预处理...")
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
            
            # 创建滑动窗口
            print("  创建滑动窗口...")
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, config['past_hours'], config['future_hours'], 
                hist_feats, fcst_feats, no_hist_power
            )
            
            if X_fcst is not None:
                print(f"  数据形状: X_hist={X_hist.shape}, X_fcst={X_fcst.shape}, y={y.shape}")
            else:
                print(f"  数据形状: X_hist={X_hist.shape}, X_fcst=None, y={y.shape}")
            
            # 按时间顺序分割数据
            total_samples = len(X_hist)
            indices = np.arange(total_samples)
            
            train_size = int(total_samples * config['train_ratio'])
            val_size = int(total_samples * config['val_ratio'])
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            # 分割数据
            X_hist_train, y_train = X_hist[train_idx], y[train_idx]
            X_hist_val, y_val = X_hist[val_idx], y[val_idx]
            X_hist_test, y_test = X_hist[test_idx], y[test_idx]
            
            # 处理X_fcst（可能为None）
            if X_fcst is not None:
                X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
            else:
                X_fcst_train, X_fcst_val, X_fcst_test = None, None, None
            
            # 分割hours和dates
            train_hours = np.array([hours[i] for i in train_idx])
            val_hours = np.array([hours[i] for i in val_idx])
            test_hours = np.array([hours[i] for i in test_idx])
            test_dates = [dates[i] for i in test_idx]
            
            train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
            val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
            test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            # 训练模型
            print("  训练模型...")
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            
            training_time = time.time() - start_time
            
            # 保存结果
            # 从experiment_name中提取特征组合信息
            # experiment_name格式: {model}_{complexity}_{feat_name}_{te_suffix}
            parts = exp_name.split('_')
            if len(parts) >= 3:
                # 提取特征组合部分（model_complexity之后，TE/noTE之前）
                feat_parts = parts[2:-1]  # 去掉model, complexity和最后的TE/noTE
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
            
            # 立即追加到CSV文件
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"  [已保存] 结果已追加到CSV (实验{idx}/160)")
            
        except Exception as e:
            print(f"  [ERROR] 实验失败: {str(e)}")
            # 记录失败的实验
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
            
            # 失败的实验也追加到CSV
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
            continue
    
    # 打印最终统计
    if results:
        print(f"\n{'='*80}")
        print("所有实验完成！最终统计:")
        print(f"{'='*80}")
        
        # 重新读取CSV文件
        final_df = pd.read_csv(output_file)
        
        print(f"结果已保存到: {output_file}")
        print(f"成功实验数: {final_df['rmse'].notna().sum()}/{len(final_df)}")
        
        # 打印最佳性能
        print(f"\n{'='*80}")
        print("Top 10 最佳RMSE:")
        print(f"{'='*80}")
        top_10 = final_df.nsmallest(10, 'rmse')[['experiment_name', 'rmse', 'mae']]
        print(top_10.to_string(index=False))
        
        return True
    else:
        print("没有成功的实验")
        return False

if __name__ == "__main__":
    success = run_all_experiments()
    if success:
        print("\n[SUCCESS] 所有160次实验完成！")
    else:
        print("\n[FAILED] 实验失败！")
        sys.exit(1)
