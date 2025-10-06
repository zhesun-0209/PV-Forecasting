# PV-Forecasting

Solar photovoltaic power forecasting system - Multi-model comparison study based on deep learning and machine learning

## 📋 Project Overview

This project implements various machine learning and deep learning models for solar PV power forecasting, including traditional ML models (Random Forest, XGBoost, LightGBM, Linear Regression) and deep learning models (LSTM, GRU, Transformer, TCN).

## 🎯 Key Features

- **Multi-Model Support**: 8 models including LSTM, GRU, Transformer, TCN, RF, XGBoost, LightGBM, Linear Regression
- **Feature Engineering**: Historical power (PV), historical weather (HW), numerical weather prediction (NWP), actual weather (NWP+)
- **Automated Experiments**: 284 experiments with batch configuration and execution
- **Comprehensive Evaluation**: RMSE, MAE, R², NRMSE, sMAPE metrics

## 📁 Project Structure

```
PV-Forecasting/
├── data/                   # Data processing module
│   ├── data_utils.py      # Preprocessing, feature engineering
│   └── Project1140.csv    # Raw dataset
│
├── models/                 # Model definitions
│   ├── ml_models.py       # ML models (RF, XGB, LGBM, Linear)
│   ├── rnn_models.py      # RNN models (LSTM, GRU)
│   ├── transformer.py     # Transformer model
│   └── tcn.py             # TCN model
│
├── train/                  # Training module
│   ├── train_dl.py        # DL training pipeline
│   ├── train_ml.py        # ML training pipeline
│   └── train_utils.py     # Training utilities
│
├── eval/                   # Evaluation module
│   ├── eval_utils.py      # Evaluation utilities
│   ├── excel_utils.py     # Excel export
│   └── metrics_utils.py   # Metrics calculation
│
├── utils/                  # Utilities
│   └── gpu_utils.py       # GPU utilities
│
├── config/                 # Configuration files
│   ├── ablation/          # Ablation study configs (229 files)
│   └── projects/1140/     # Project configs (285 files)
│
├── run_all_experiments.py  # Main experiment script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run all 284 experiments
python run_all_experiments.py
```

## 📊 Feature Definitions

- **PV**: Historical photovoltaic power data
- **HW**: Historical weather data (temperature, humidity, irradiance, etc.)
- **NWP**: Numerical weather prediction (forecast weather)
- **NWP+**: Actual weather data (ideal scenario)

## 🔧 Model Configurations

### Supported Models

**Deep Learning Models:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer
- TCN (Temporal Convolutional Network)

**Machine Learning Models:**
- Random Forest
- XGBoost
- LightGBM
- Linear Regression

### Complexity Levels

- **Low Complexity**: Fast experiments, limited resources
- **High Complexity**: Higher accuracy, more resources
- **Linear Model**: No complexity parameter (single configuration)

## 📈 Experiment Design

Total experiments: **284**

**Breakdown:**
- DL models (LSTM, GRU, Transformer, TCN): 4 × 40 = 160
- ML models (RF, XGB, LGBM): 3 × 40 = 120
- Linear model: 4 (NWP/NWP+ with TE options)

**Feature Combinations:**
1. PV (historical power only)
2. PV + HW (historical power + historical weather)
3. PV + NWP (historical power + forecast weather)
4. PV + NWP+ (historical power + actual weather)
5. NWP (forecast weather only)
6. NWP+ (actual weather only)

**Variables:**
- Lookback windows: 24h, 72h (not applicable for NWP-only and Linear)
- Time encoding: Enabled / Disabled
- Model complexity: Low / High (not applicable for Linear)

## 🎓 Key Findings

1. **LSTM performs best**: RMSE of 6.01 in PV+NWP scenario
2. **Feature combination matters**: PV+NWP outperforms single features
3. **NWP vs NWP+**: Forecast weather sometimes outperforms actual weather (smoothness advantage)
4. **Model selection**: For medium-scale data, RNNs outperform Transformer/TCN

## 📖 Documentation

All code is documented in English with clear comments and docstrings.

## 🔬 Research Background

This project is designed for solar PV power forecasting research, comparing different models and feature combinations to provide practical references.

## 📝 Citation

If you use this code, please cite:

```
@misc{pv-forecasting-2025,
  title={PV-Forecasting: A Multi-Model Comparison for Solar Power Prediction},
  author={Zhe Sun},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zhesun-0209/PV-Forecasting}
}
```

## 📧 Contact

For questions or suggestions, please submit an Issue or Pull Request.

## 📄 License

MIT License

---

**Note**: The dataset `Project1140.csv` is included in the project. To use your own data, please refer to the data format specifications in the code.
