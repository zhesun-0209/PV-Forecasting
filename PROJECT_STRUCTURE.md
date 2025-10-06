# 太阳能光伏发电预测项目结构

## 📁 项目目录结构

```
Solar Prediction/
│
├── data/                           # 数据处理模块
│   ├── data_utils.py              # 数据预处理、特征工程、滑动窗口
│   └── Project1140.csv            # 原始数据集
│
├── models/                         # 模型定义
│   ├── ml_models.py               # 传统机器学习模型 (RF, XGBoost, etc.)
│   ├── rnn_models.py              # RNN模型 (LSTM, GRU)
│   ├── transformer.py             # Transformer模型
│   └── tcn.py                     # TCN (时间卷积网络)
│
├── train/                          # 训练模块
│   ├── train_dl.py                # 深度学习训练流程
│   ├── train_ml.py                # 机器学习训练流程
│   └── train_utils.py             # 训练工具函数 (optimizer, scheduler)
│
├── eval/                           # 评估模块
│   ├── eval_utils.py              # 评估工具
│   ├── excel_utils.py             # Excel导出工具
│   └── metrics_utils.py           # 指标计算 (RMSE, MAE, R2)
│
├── utils/                          # 通用工具
│   ├── __init__.py
│   └── gpu_utils.py               # GPU工具
│
├── config/                         # 配置文件
│   └── projects/
│       └── 1140/                  # Project1140的配置文件
│           └── *.yaml             # 各种实验配置
│
├── run_all_experiments.py         # 主实验脚本 (160次实验)
│
├── requirements.txt                # Python依赖
│
├── corrected_improvement_strategy.md  # 模型优化策略文档
│
└── 实验结果文件/
    ├── All_Models_4Scenarios_Comparison.xlsx           # 4模型×4场景对比
    ├── GRU_NWP_vs_NWP_plus_comparison.xlsx            # GRU NWP对比
    ├── Optimization_Strategies_Comparison.csv          # 优化策略测试结果
    ├── GRU_4scenarios_comparison.png                   # GRU对比图
    ├── GRU_NWP_vs_NWP_plus_direct_comparison.png      # GRU直接对比图
    ├── Model_Performance_Comprehensive_Analysis.png    # 模型性能综合分析
    └── Transformer_TCN_Improvement_Roadmap.png         # 改进路线图
```

---

## 🎯 核心功能模块

### **1. 数据处理 (`data/data_utils.py`)**

#### 特征定义：
- **PV**: 历史光伏功率 (`power`)
- **HW**: 历史天气 (温度、湿度、辐照度等，后缀无`_pred`)
- **NWP**: 预测天气 (后缀`_pred`)
- **NWP+**: 实际天气作为理想NWP (后缀无`_pred`，作为forecast特征)

#### 主要函数：
- `get_weather_features_by_category()`: 获取不同类别的天气特征
- `preprocess_features()`: 数据预处理、归一化、日期过滤
- `create_sliding_windows()`: 创建滑动窗口 (24小时lookback → 24小时预测)

---

### **2. 模型定义 (`models/`)**

#### 支持的模型：
```python
传统ML:
- RandomForest
- XGBoost
- LightGBM
- CatBoost

深度学习:
- LSTM (2种复杂度: Low/High)
- GRU (2种复杂度: Low/High)
- Transformer (2种复杂度: Low/High)
- TCN (2种复杂度: Low/High)
```

#### 复杂度配置：
```python
Low Complexity:
- d_model: 16, hidden_dim: 8, num_layers: 1
- epochs: 20, batch_size: 64, lr: 0.001

High Complexity:
- d_model: 32, hidden_dim: 16, num_layers: 2
- epochs: 50, batch_size: 64, lr: 0.001
```

---

### **3. 训练模块 (`train/`)**

#### 训练流程：
1. 数据加载与预处理
2. 模型初始化
3. 训练循环 (含Early Stopping)
4. 验证与测试
5. 指标计算 (RMSE, MAE, R2)

#### 优化器与正则化：
- Optimizer: AdamW
- Learning Rate: 0.001
- Weight Decay: 1e-4 (L2正则化)
- Early Stopping: patience=10, min_delta=0.001

---

### **4. 评估模块 (`eval/`)**

#### 评估指标：
```python
RMSE: 均方根误差 (主要指标)
MAE:  平均绝对误差
R2:   决定系数
```

#### 数据范围：
- 所有指标基于逆变换后的真实值 (0-100 容量因子百分比)

---

## 🧪 实验设置

### **完整实验矩阵 (160次实验)**

```
模型 (8种):
- LSTM_low, LSTM_high
- GRU_low, GRU_high
- Transformer_low, Transformer_high
- TCN_low, TCN_high

特征组合 (10种):
1. PV (历史功率)
2. HW (历史天气)
3. NWP (预测天气)
4. NWP+ (实际天气)
5. PV + HW
6. PV + NWP
7. PV + NWP+
8. HW + NWP
9. HW + NWP+
10. PV + HW + NWP

Lookback时长 (2种):
- 24小时
- 72小时

总计: 8 × 10 × 2 = 160 次实验
```

---

## 📊 实验结果总结

### **最佳配置：**
```
模型: LSTM (High Complexity)
特征: PV + NWP
Lookback: 24小时
Time Encoding: 启用

RMSE: 6.0097
MAE:  3.7227
R2:   0.9290
```

### **各模型性能对比（PV+NWP, 24h）：**
```
LSTM:        6.01  ✅ 最好
GRU:         6.15  (差2.3%)
Transformer: 6.57  (差9.3%)
TCN:         8.27  (差37.6%)
```

### **特征组合对比（LSTM High, 24h）：**
```
PV + NWP:    6.01  ✅ 最好
PV + NWP+:   6.23  (差3.7%, 理想场景反而更差)
PV only:     6.54  (差8.8%)
NWP only:    7.89  (差31.3%)
```

---

## 🔍 关键发现

### **1. NWP+ 性能不如 NWP**

**原因分析：**
- NWP (预测天气) 更平滑、异常值少 (IQR异常值占26.2%)
- NWP+ (实际天气) 波动大、异常值多 (IQR异常值占100%)
- 模型更容易学习平滑的NWP模式

**结论：** 这是有效的科学发现，说明"平滑但有偏的预测"可能比"准确但波动的真实值"更适合作为特征。

---

### **2. High Complexity > Low Complexity**

**优化历程：**
- 初始: High性能差于Low (过拟合)
- 优化策略:
  - 减小模型复杂度 (layers 6→2, hidden 64→16)
  - 统一训练参数 (batch_size=64, lr=0.001)
  - 增加L2正则化 (weight_decay=1e-4)
  - 调整epochs (Low=20, High=50)
- 结果: High性能超越Low

**结论：** 通过合理的正则化和训练策略，更复杂的模型可以实现更好的性能。

---

### **3. Transformer/TCN 不如 LSTM/GRU**

**尝试的优化策略：**
1. ❌ 减小模型复杂度 → 性能崩溃 (欠拟合)
2. ✅ 优化训练超参数 → TCN轻微改进 (3.5%)
3. ⚠️ 增大模型复杂度 → 未测试

**原因分析：**
- 数据规模小 (24k样本)
- 序列短 (24小时)
- RNN的归纳偏置更适合时序任务
- Transformer需要更多数据和更长序列才能发挥优势

**建议：**
- 对于中小规模光伏预测任务，优先使用LSTM/GRU
- Transformer/TCN更适合大规模、长序列任务

---

## 🚀 如何运行

### **运行完整实验 (160次):**
```bash
python run_all_experiments.py
```

### **单独测试特定配置:**
```python
from train.train_dl import train_dl_model
from data.data_utils import preprocess_features, create_sliding_windows

config = {
    'model': 'LSTM',
    'model_complexity': 'high',
    'past_hours': 24,
    'use_pv': True,
    'use_forecast': True,
    'use_ideal_nwp': False,
    # ... 其他配置
}

# 运行训练
model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
print(f"RMSE: {metrics['rmse']:.4f}")
```

---

## 📦 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.x
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, seaborn

---

## 📝 配置文件说明

配置文件位于 `config/projects/1140/`，包含各种实验的YAML配置。

示例配置：
```yaml
model: LSTM
model_complexity: high
past_hours: 24
future_hours: 24
use_pv: true
use_forecast: true
use_ideal_nwp: false
use_time_encoding: true

train_params:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 10
```

---

## 📈 结果文件

所有实验结果保存为Excel/CSV格式，包含：
- 各模型在不同场景下的RMSE/MAE/R2
- 训练时间
- 最佳epoch
- 详细预测结果（部分实验）

---

## ✅ 清理记录

**已删除的临时文件：**
- `compare_nwp_single_experiment.py`
- `compare_gru_nwp_experiments.py`
- `compare_4models_4scenarios.py`
- `test_optimization_strategies.py`
- `plot_gru_nwp_vs_nwp_plus.py`
- `visualize_model_comparison.py`
- `visualize_improvement_roadmap.py`
- `analysis_why_transformer_tcn_underperform.md`
- `improve_transformer_tcn_strategy.md`
- `config/ablation/` (229个YAML文件)
- 所有 `__pycache__` 目录

**保留的核心文件：**
- 核心代码模块 (data, models, train, eval, utils)
- 主实验脚本 (run_all_experiments.py)
- 实验结果文件 (Excel, CSV, PNG)
- 最新策略文档 (corrected_improvement_strategy.md)

---

## 📧 联系方式

Project: Solar Power Prediction
Dataset: Project1140
Date: 2025-10-06

