# Colab 多电站批量实验运行指南

本指南帮助您在 Google Colab 上运行100个电站的光伏功率预测实验。

## 📋 前置要求

- Google Colab账号
- 100个数据集CSV文件
- 数据集格式要求：包含 Year, Month, Day, Hour, 功率字段, 天气字段等

---

## 🚀 快速开始

### 1. 在Colab中克隆项目

```python
# 克隆GitHub仓库
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# 查看项目结构
!ls -la
```

### 2. 安装依赖

```python
# 安装必要的Python包
!pip install -r requirements.txt

# 安装GPU加速的LightGBM和XGBoost
!pip install lightgbm xgboost torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 检查GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## 📁 准备数据集

### 方法A: 上传到Colab

```python
from google.colab import files
import shutil
import os

# 创建data目录（如果不存在）
os.makedirs('data', exist_ok=True)

# 上传文件（可以多选）
print("请选择您的100个CSV文件...")
uploaded = files.upload()

# 移动文件到data目录
for filename in uploaded.keys():
    shutil.move(filename, f'data/{filename}')
    print(f'已移动: {filename}')
```

### 方法B: 从Google Drive挂载

```python
from google.colab import drive
drive.mount('/content/drive')

# 从Drive复制数据到data目录
!cp /content/drive/MyDrive/your_dataset_folder/*.csv data/
!ls data/
```

### 数据集命名建议

建议命名格式：`Project{ID}.csv` 或 `Plant{ID}.csv`

例如：
- `Project1001.csv`
- `Project1002.csv`
- ...
- `Project1100.csv`

---

## ⚙️ 批量创建配置文件

运行我们提供的批量配置生成脚本：

```python
!python batch_create_configs.py
```

这个脚本会：
1. 自动扫描 `data/` 目录下的所有CSV文件
2. 为每个数据集创建对应的配置文件
3. 保存到 `config/plants/Plant{ID}.yaml`

---

## 🎯 运行实验

### 单个电站实验（测试用）

```python
# 运行单个电站的284个实验
!python run_all_experiments.py
```

### 多电站批量实验（推荐）

```python
# 运行所有100个电站的实验
!python run_experiments_multi_plant.py
```

### 指定特定电站

```python
# 只运行特定的几个电站
!python run_experiments_multi_plant.py --plants 1001 1002 1003
```

---

## 📊 监控进度

### 实时监控

在另一个代码单元中运行：

```python
!python monitor_progress.py
```

### 检查进度

```python
!python check_progress.py
```

---

## 💾 保存结果

### 下载所有结果

```python
from google.colab import files
import glob

# 打包所有结果
!zip -r all_results.zip results/ *.csv

# 下载
files.download('all_results.zip')
```

### 保存到Google Drive

```python
from google.colab import drive
import shutil

drive.mount('/content/drive')

# 复制结果到Drive
!mkdir -p /content/drive/MyDrive/PV_Forecasting_Results
!cp -r results/ /content/drive/MyDrive/PV_Forecasting_Results/
!cp *.csv /content/drive/MyDrive/PV_Forecasting_Results/

print("✓ 结果已保存到Google Drive!")
```

---

## ⚡ 性能优化建议

### 1. 使用GPU运行时

- 在Colab中选择：`运行时` > `更改运行时类型` > `硬件加速器` > `GPU`
- 推荐使用T4或A100 GPU

### 2. 分批次运行

如果100个电站太多，可以分批：

```python
# 第1批：电站1-25
!python run_experiments_multi_plant.py --start 1 --end 25

# 第2批：电站26-50
!python run_experiments_multi_plant.py --start 26 --end 50

# ...依此类推
```

### 3. 预计运行时间

- 单个电站284个实验：约2-3小时（GPU）
- 100个电站：约200-300小时
- **建议分批运行，或使用多个Colab实例并行**

---

## 🔧 常见问题

### Q1: Colab会话超时怎么办？

**解决方案**：使用Resume功能

```python
# 脚本会自动检测已完成的实验并跳过
# 重新运行即可从断点继续
!python run_experiments_multi_plant.py
```

### Q2: 内存不足？

**解决方案**：

```python
# 清理内存
import gc
gc.collect()

# 或者在run_experiments_multi_plant.py中设置
# 每完成一个电站就清理一次内存
```

### Q3: 如何并行运行多个电站？

**方案1**: 使用多个Colab Notebook

每个Notebook运行不同批次：
- Notebook 1: 电站1-20
- Notebook 2: 电站21-40
- Notebook 3: 电站41-60
- ...

**方案2**: 使用Colab Pro+

获得更长的运行时间和更好的GPU。

---

## 📝 完整示例脚本

### 一键运行脚本

创建一个新的代码单元：

```python
# === 1. 环境准备 ===
print("Step 1: 克隆项目...")
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

print("Step 2: 安装依赖...")
!pip install -q -r requirements.txt

print("Step 3: 检查GPU...")
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# === 2. 准备数据 ===
print("\nStep 4: 挂载Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

print("Step 5: 复制数据集...")
!mkdir -p data
!cp /content/drive/MyDrive/你的数据集文件夹/*.csv data/
print(f"数据集数量: {len(!ls data/*.csv)}")

# === 3. 生成配置 ===
print("\nStep 6: 批量生成配置文件...")
!python batch_create_configs.py

# === 4. 运行实验 ===
print("\nStep 7: 开始运行实验...")
!python run_experiments_multi_plant.py

# === 5. 保存结果 ===
print("\nStep 8: 保存结果到Drive...")
!mkdir -p /content/drive/MyDrive/PV_Results
!cp -r results/ /content/drive/MyDrive/PV_Results/
!cp *.csv /content/drive/MyDrive/PV_Results/

print("\n✓ 全部完成！")
```

---

## 📚 高级用法

### 自定义配置

如果需要修改某些电站的配置：

```python
# 编辑特定电站配置
!nano config/plants/Plant1001.yaml

# 或使用Python脚本批量修改
import yaml
import glob

for config_file in glob.glob('config/plants/*.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置（例如：更改时间范围）
    config['start_date'] = '2023-01-01'
    config['end_date'] = '2023-12-31'
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
```

### 只运行特定模型

修改 `run_all_experiments.py` 中的模型列表：

```python
# 只运行深度学习模型
dl_models = ['LSTM', 'GRU']  # 移除Transformer和TCN

# 只运行机器学习模型
ml_models = ['LGBM', 'XGB']  # 移除RF
```

---

## 🎓 最佳实践

1. **先测试一个电站**：确保流程正确
2. **定期保存结果**：每完成一批就保存到Drive
3. **监控GPU使用**：避免资源浪费
4. **使用Resume功能**：从中断处继续
5. **分批运行**：避免超长运行导致会话超时

---

## 📞 获取帮助

如果遇到问题：

1. 查看错误日志
2. 检查数据格式是否正确
3. 确认GPU可用
4. 查看GitHub Issues

---

**祝实验顺利！🚀**

