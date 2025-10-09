#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时监控实验进度"""

import pandas as pd
import os
import time
from datetime import datetime
import glob

def monitor_progress():
    """监控实验进度"""
    print("=" * 80)
    print("实验进度监控")
    print("=" * 80)
    
    # 查找结果文件
    result_files = glob.glob("results_*_*.csv")
    if not result_files:
        print("未找到结果文件，实验可能还未开始...")
        return
    
    # 使用最新的文件
    result_file = max(result_files, key=os.path.getmtime)
    print(f"监控文件: {result_file}")
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
                    bar_length = int(pct / 5)
                    bar = '█' * bar_length + '░' * (20 - bar_length)
                    print(f"{model:12s}: {bar} {actual:3d}/{exp:3d} ({pct:5.1f}%)")
        
        # 最近完成的实验
        print("\n最近完成的5个实验:")
        print(df[['experiment_name', 'model', 'mae', 'rmse', 'r2']].tail(5).to_string(index=False))
        
        # 性能统计
        if len(df) > 0 and 'mae' in df.columns:
            print("\n当前最佳结果 (MAE):")
            best_idx = df['mae'].idxmin()
            best = df.loc[best_idx]
            print(f"  实验: {best['experiment_name']}")
            print(f"  MAE: {best['mae']:.4f}, RMSE: {best['rmse']:.4f}, R²: {best['r2']:.4f}")
        
        # 检查重复
        if 'experiment_name' in df.columns:
            dups = df[df.duplicated('experiment_name', keep=False)]
            if len(dups) > 0:
                print(f"\n⚠️ 警告: 发现 {len(dups)} 个重复实验!")
                print(dups['experiment_name'].unique()[:5])
            else:
                print("\n✅ 无重复实验")
        
        # 预计完成时间
        if 'train_time_sec' in df.columns and completed > 0:
            avg_time = df['train_time_sec'].mean()
            remaining = total - completed
            est_time = remaining * avg_time
            est_hours = est_time / 3600
            print(f"\n预计剩余时间: {est_hours:.1f} 小时 (平均每个实验 {avg_time:.1f}秒)")
        
    except Exception as e:
        print(f"读取文件出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            monitor_progress()
            print("\n按 Ctrl+C 退出监控...")
            time.sleep(10)  # 每10秒刷新一次
    except KeyboardInterrupt:
        print("\n\n监控已停止")

