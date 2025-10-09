#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量创建配置文件
自动扫描data目录下的所有CSV文件，为每个数据集生成对应的配置文件
"""

import os
import glob
import yaml
import pandas as pd
from pathlib import Path
import re

def extract_plant_id(filename):
    """
    从文件名中提取电站ID
    支持格式：Project1140.csv, Plant1140.csv, 1140.csv等
    """
    basename = os.path.basename(filename)
    
    # 尝试多种模式
    patterns = [
        r'Project(\d+)',
        r'Plant(\d+)',
        r'plant(\d+)',
        r'project(\d+)',
        r'^(\d+)',  # 纯数字开头
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # 如果都不匹配，使用文件名（去除扩展名）
    return basename.replace('.csv', '').replace('.CSV', '')

def detect_date_range(csv_file):
    """
    自动检测数据集的时间范围
    """
    try:
        # 读取CSV文件（只读取前几行和最后几行来节省时间）
        df_head = pd.read_csv(csv_file, nrows=10)
        df = pd.read_csv(csv_file)
        
        # 尝试构建日期时间
        if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
            df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            start_date = df['Datetime'].min().strftime('%Y-%m-%d')
            end_date = df['Datetime'].max().strftime('%Y-%m-%d')
            return start_date, end_date, len(df)
        else:
            print(f"  ⚠️  警告: {csv_file} 缺少必要的时间列，使用默认日期")
            return '2022-01-01', '2024-09-28', len(df)
    except Exception as e:
        print(f"  ⚠️  读取 {csv_file} 时出错: {str(e)}")
        return '2022-01-01', '2024-09-28', 0

def create_plant_config(csv_file, template_path='config/plant_template.yaml'):
    """
    为单个数据集创建配置文件
    """
    # 读取模板
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取电站ID
    plant_id = extract_plant_id(csv_file)
    
    # 检测日期范围
    start_date, end_date, data_length = detect_date_range(csv_file)
    
    # 更新配置
    config['plant_id'] = str(plant_id)
    config['plant_name'] = f"Project {plant_id}"
    config['data_path'] = csv_file
    config['start_date'] = start_date
    config['end_date'] = end_date
    
    # 保留模板中的其他默认设置
    # shuffle_split默认为False (Sequential划分)
    if 'shuffle_split' not in config:
        config['shuffle_split'] = False
    if 'random_seed' not in config:
        config['random_seed'] = 42
    if 'past_hours' not in config:
        config['past_hours'] = 24
    
    return plant_id, config, data_length

def batch_create_configs(data_dir='data', output_dir='config/plants', template='config/plant_template.yaml'):
    """
    批量创建配置文件
    """
    print("="*80)
    print("批量创建配置文件")
    print("="*80)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv')) + glob.glob(os.path.join(data_dir, '*.CSV'))
    
    if not csv_files:
        print(f"\n❌ 错误: 在 {data_dir} 目录下没有找到CSV文件")
        print(f"请确保数据集文件已放置在 {data_dir} 目录中")
        return
    
    print(f"\n找到 {len(csv_files)} 个数据集文件")
    print("-"*80)
    
    created_configs = []
    skipped_configs = []
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"\n[{i}/{len(csv_files)}] 处理: {filename}")
        
        try:
            # 创建配置
            plant_id, config, data_length = create_plant_config(csv_file, template)
            
            # 保存配置文件
            config_file = os.path.join(output_dir, f'Plant{plant_id}.yaml')
            
            # 检查是否已存在
            if os.path.exists(config_file):
                print(f"  ⚠️  配置文件已存在，跳过: {config_file}")
                skipped_configs.append(plant_id)
                continue
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"  ✓ 创建配置: {config_file}")
            print(f"    电站ID: {plant_id}")
            print(f"    数据范围: {config['start_date']} 至 {config['end_date']}")
            print(f"    数据量: {data_length} 条记录")
            
            created_configs.append(plant_id)
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*80)
    print("配置文件创建完成")
    print("="*80)
    print(f"✓ 新创建: {len(created_configs)} 个配置文件")
    print(f"⚠️ 已跳过: {len(skipped_configs)} 个配置文件（已存在）")
    print(f"📁 配置目录: {output_dir}")
    
    if created_configs:
        print(f"\n创建的电站ID: {', '.join(created_configs[:10])}" + 
              (f" ... (共{len(created_configs)}个)" if len(created_configs) > 10 else ""))
    
    print("\n下一步:")
    print("  1. 检查配置文件: ls config/plants/")
    print("  2. 运行单个电站测试: python run_all_experiments.py")
    print("  3. 运行所有电站: python run_experiments_multi_plant.py")
    print("="*80)

def verify_configs(config_dir='config/plants'):
    """
    验证所有配置文件
    """
    print("\n验证配置文件...")
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))
    
    valid_count = 0
    invalid_count = 0
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查必要字段
            required_fields = ['plant_id', 'data_path', 'start_date', 'end_date']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                print(f"  ⚠️  {os.path.basename(config_file)}: 缺少字段 {missing_fields}")
                invalid_count += 1
            else:
                # 检查数据文件是否存在
                if not os.path.exists(config['data_path']):
                    print(f"  ⚠️  {os.path.basename(config_file)}: 数据文件不存在 {config['data_path']}")
                    invalid_count += 1
                else:
                    valid_count += 1
        except Exception as e:
            print(f"  ❌ {os.path.basename(config_file)}: {str(e)}")
            invalid_count += 1
    
    print(f"\n验证结果: ✓ {valid_count} 个有效, ⚠️ {invalid_count} 个无效")
    return valid_count, invalid_count

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量创建配置文件')
    parser.add_argument('--data-dir', default='data', help='数据集目录 (默认: data)')
    parser.add_argument('--output-dir', default='config/plants', help='配置输出目录 (默认: config/plants)')
    parser.add_argument('--template', default='config/plant_template.yaml', help='配置模板文件')
    parser.add_argument('--verify', action='store_true', help='验证已创建的配置文件')
    parser.add_argument('--force', action='store_true', help='强制覆盖已存在的配置文件')
    
    args = parser.parse_args()
    
    # 检查模板文件是否存在
    if not os.path.exists(args.template):
        print(f"❌ 错误: 模板文件不存在: {args.template}")
        exit(1)
    
    # 如果只是验证
    if args.verify:
        verify_configs(args.output_dir)
        exit(0)
    
    # 批量创建配置
    batch_create_configs(args.data_dir, args.output_dir, args.template)
    
    # 验证创建的配置
    print("\n")
    verify_configs(args.output_dir)

