# PV-Forecasting

光伏发电功率预测系统 - 基于深度学习和机器学习的多模型对比研究

## 📋 项目简介

本项目实现了多种机器学习和深度学习模型用于太阳能光伏发电功率预测，包括传统机器学习模型（Random Forest, XGBoost, LightGBM, CatBoost）和深度学习模型（LSTM, GRU, Transformer, TCN）。

## 🎯 主要特性

- **多模型支持**: LSTM, GRU, Transformer, TCN, Random Forest, XGBoost等
- **特征工程**: 支持历史功率、历史天气、预测天气等多种特征组合
- **自动化实验**: 支持批量实验配置和自动运行
- **完整评估**: 提供RMSE, MAE, R2等多种评估指标

## 📁 项目结构

```
PV-Forecasting/
├── data/                   # 数据处理模块
│   ├── data_utils.py      # 数据预处理、特征工程
│   └── Project1140.csv    # 原始数据集
│
├── models/                 # 模型定义
│   ├── ml_models.py       # 机器学习模型
│   ├── rnn_models.py      # RNN模型 (LSTM, GRU)
│   ├── transformer.py     # Transformer模型
│   └── tcn.py             # TCN模型
│
├── train/                  # 训练模块
│   ├── train_dl.py        # 深度学习训练
│   ├── train_ml.py        # 机器学习训练
│   └── train_utils.py     # 训练工具
│
├── eval/                   # 评估模块
│   ├── eval_utils.py      # 评估工具
│   ├── excel_utils.py     # Excel导出
│   └── metrics_utils.py   # 指标计算
│
├── utils/                  # 工具模块
│   └── gpu_utils.py       # GPU工具
│
├── config/                 # 配置文件
│   └── projects/1140/     # 项目配置
│
├── run_all_experiments.py  # 主实验脚本
├── requirements.txt        # 依赖文件
└── README.md              # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 运行所有实验（160次）
python run_all_experiments.py
```

## 📊 特征定义

- **PV**: 历史光伏功率数据
- **HW**: 历史天气数据（温度、湿度、辐照度等）
- **NWP**: 数值天气预报数据（预测天气）
- **NWP+**: 实际天气数据（理想场景）

## 🔧 模型配置

### 支持的模型

**深度学习模型：**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer
- TCN (Temporal Convolutional Network)

**机器学习模型：**
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### 复杂度配置

- **Low Complexity**: 适合快速实验和资源受限场景
- **High Complexity**: 适合追求更高精度

## 📈 实验配置

项目支持以下实验变量：
- 模型类型：8种（4种DL × 2复杂度）
- 特征组合：10种
- Lookback时长：24小时 / 72小时
- 时间编码：启用/禁用

总实验次数：**160次**

## 🎓 主要发现

1. **LSTM表现最佳**: 在PV+NWP场景下RMSE达到6.01
2. **特征组合重要**: PV+NWP组合优于单一特征
3. **NWP vs NWP+**: 预测天气有时优于实际天气（平滑性优势）
4. **模型选择**: 对于中小规模数据，RNN优于Transformer/TCN

详细分析请参考项目文档。

## 📖 文档

- `PROJECT_STRUCTURE.md` - 详细项目结构说明
- `corrected_improvement_strategy.md` - 模型优化策略
- `CLEANUP_REPORT.md` - 代码清理报告

## 🔬 研究背景

本项目用于太阳能光伏发电功率预测研究，旨在对比不同模型和特征组合在光伏预测任务中的表现，为实际应用提供参考。

## 📝 引用

如果您使用本项目代码，请引用：

```
@misc{pv-forecasting-2025,
  title={PV-Forecasting: A Multi-Model Comparison for Solar Power Prediction},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zhesun-0209/PV-Forecasting}
}
```

## 📧 联系方式

如有问题或建议，请提交Issue或Pull Request。

## 📄 许可证

MIT License

---

**注意**: 数据文件`Project1140.csv`较大，已包含在项目中。如需使用自己的数据，请参考数据格式说明。

