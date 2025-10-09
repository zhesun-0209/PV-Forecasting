#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的实验进度检查"""

import pandas as pd
import os
import glob
from datetime import datetime

def check_progress():
    """检查实验进度"""
    print("=" * 80)
    print("实验进度检查")
    print("=" * 80)
    
    # 查找结果文件
    result_files = glob.glob("results_*_*.csv")
    if not result_files:
        print("未找到结果文件，实验可能还未开始或第一个实验还在运行中...")
        return
    
    # 使用最新的文件
    result_file = max(result_files, key=os.path.getmtime)
    print(f"结果文件: {result_file}")
    print(f"最后更新: {datetime.fromtimestamp(os.path.getmtime(result_file)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 读取结果
    try:
        df = pd.read_csv(result_file)
        completed = len(df)
        total = 284
        progress = (completed / total) * 100
        
        print(f"\n进度: {completed}/{total} ({progress:.1f}%)")
        print("=" * 80)
        
        # 按模型统计
        if 'model' in df.columns:
            print("\n各模型完成情况:")
            model_counts = df['model'].value_counts().sort_index()
            expected = {'LSTM': 40, 'GRU': 40, 'Transformer': 40, 'TCN': 40, 
                       'RF': 60, 'XGB': 60, 'LGBM': 60, 'Linear': 4}
            
            for model in ['LSTM', 'GRU', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear']:
                actual = model_counts.get(model, 0)
                exp = expected.get(model, 0)
                if exp > 0:
                    pct = (actual / exp) * 100
                    print(f"{model:12s}: {actual:3d}/{exp:3d} ({pct:5.1f}%)")
        
        # 检查test_samples是否一致
        if 'test_samples' in df.columns:
            unique_samples = df['test_samples'].unique()
            print(f"\n测试样本数: {unique_samples}")
            if len(unique_samples) > 1:
                print("警告: 不同实验的测试样本数不一致!")
                print("\n各模型的测试样本数:")
                print(df.groupby('model')['test_samples'].unique().to_dict())
        
        # 最近完成的实验
        print("\n最近完成的5个实验:")
        cols_to_show = ['experiment_name', 'model', 'test_samples', 'mae', 'rmse']
        cols_available = [c for c in cols_to_show if c in df.columns]
        print(df[cols_available].tail(5).to_string(index=False))
        
        # 检查重复
        if 'experiment_name' in df.columns:
            dups = df[df.duplicated('experiment_name', keep=False)]
            if len(dups) > 0:
                print(f"\n[警告] 发现 {len(dups)} 个重复实验!")
            else:
                print("\n[OK] 无重复实验")
        
        # 预计完成时间
        if 'train_time_sec' in df.columns and completed > 0:
            avg_time = df['train_time_sec'].mean()
            remaining = total - completed
            est_time = remaining * avg_time
            est_hours = est_time / 3600
            print(f"\n预计剩余时间: {est_hours:.1f} 小时")
        
    except Exception as e:
        print(f"读取文件出错: {e}")

if __name__ == "__main__":
    check_progress()

