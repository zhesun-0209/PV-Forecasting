# Solar PV Power Forecasting - Multi-Plant System

A comprehensive system for solar photovoltaic power forecasting using multiple deep learning and machine learning models, with support for batch processing multiple plants.

**Base Repository**: https://github.com/zhesun-0209/PV-Forecasting.git  
**Optimized Version**: v2.0 (2025-10-09)

### Key Improvements Over Base Repository

✅ **GPU OOM Protection**: Batch prediction and memory management to prevent GPU out-of-memory issues  
✅ **Random Shuffle**: Test set covers all seasons for robust evaluation (vs sequential split)  
✅ **Daily Average Metrics**: RMSE/MAE calculated per day then averaged (better for day-ahead forecasting)  
✅ **Flexible Lookback**: Support for 24h and 72h historical windows  
✅ **Tree Model Optimization**: High complexity models use n_estimators=100, max_depth=10  
✅ **Silent Mode**: LightGBM verbosity=-1, XGBoost verbosity=0 to suppress warnings  
✅ **Multi-Plant Support**: Unified configuration system for batch processing  
✅ **Reproducibility**: Fixed random seed (42) ensures consistent results

---

## 🎯 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Configurations  
```bash
# Automatically scan data/ and create configs for all plants
python batch_create_configs.py
```

### Step 3: Run Experiments
```bash
# Option A: Single plant (284 experiments)
python run_all_experiments.py

# Option B: All plants batch processing
python run_experiments_multi_plant.py
```

**That's it!** Each plant gets 284 experiments with standardized metrics.

---

## 📋 Features

### 🔬 Models Supported (8 Total)

**Deep Learning Models:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)  
- Transformer
- TCN (Temporal Convolutional Network)

**Machine Learning Models:**
- Random Forest (GPU-accelerated with cuML)
- XGBoost (GPU-accelerated, using device='cuda')
- LightGBM (GPU-accelerated)
- Linear Regression

### 📊 Evaluation Method

**Daily Average Metrics** (Recommended for Day-Ahead Forecasting):
- RMSE and MAE calculated per day, then averaged across all test days
- More accurately reflects daily prediction performance
- Better suited for day-ahead forecasting scenarios than flattening all hourly values

**Data Split Strategy**:
- Random shuffle with fixed seed (seed=42) for reproducibility
- Ensures test set covers all seasons and weather conditions
- More robust evaluation compared to sequential splitting

### 📊 Feature Scenarios (6 Types)

- **PV**: Historical PV power only
- **PV+HW**: PV + Historical Weather
- **PV+NWP**: PV + Numerical Weather Prediction
- **PV+NWP+**: PV + Ideal NWP (actual weather as forecast)
- **NWP**: NWP forecast only
- **NWP+**: Ideal NWP only

### 🎛️ Experiment Configuration

Each plant runs **284 experiments** covering all combinations:

| Category | Count | Details |
|----------|-------|---------|
| DL (PV-based) | 128 | 4 models × 2 complexity × 2 lookback × 2 TE × 4 scenarios |
| DL (NWP-only) | 32 | 4 models × 2 complexity × 2 TE × 2 scenarios |
| ML (PV-based) | 96 | 3 models × 2 complexity × 2 lookback × 2 TE × 4 scenarios |
| ML (NWP-only) | 24 | 3 models × 2 complexity × 2 TE × 2 scenarios |
| Linear (NWP) | 4 | 1 model × 2 TE × 2 scenarios |
| **Total** | **284** | |

**Variables:**
- **Lookback windows**: 24h (1 day), 72h (3 days) for scenarios with historical features (PV, PV+HW)
- **Time Encoding**: Enabled / Disabled  
- **Model Complexity**: Low / High (DL and ML models)

**Note**: NWP-only scenarios use past_hours=0 (no historical lookback), relying solely on weather forecasts.

---

## 📁 Project Structure

```
Solar Prediction/
├── config/
│   ├── plant_template.yaml          # Configuration template
│   └── plants/                      # Plant-specific configs (1 per plant)
│       ├── Plant1140.yaml
│       ├── Plant1141.yaml
│       └── ...
│
├── data/                            # Data files (one CSV per plant)
│   ├── Project1140.csv
│   ├── Project1141.csv
│   └── ...
│
├── models/                          # Model implementations
│   ├── ml_models.py                # ML models (RF, XGB, LGBM, Linear)
│   ├── rnn_models.py               # RNN models (LSTM, GRU)
│   ├── transformer.py              # Transformer model
│   └── tcn.py                      # TCN model
│
├── train/                          # Training pipelines
│   ├── train_dl.py                # Deep learning training
│   ├── train_ml.py                # Machine learning training
│   └── train_utils.py             # Training utilities
│
├── eval/                           # Evaluation utilities
│   ├── eval_utils.py              # Result saving
│   ├── excel_utils.py             # Excel export
│   ├── metrics_utils.py           # Metrics calculation
│   └── prediction_utils.py        # 24h-ahead extraction
│
├── utils/                          # Utilities
│   ├── gpu_utils.py               # GPU memory management
│   └── __init__.py
│
├── run_all_experiments.py          # Main script (single plant)
├── run_experiments_multi_plant.py  # Batch script (all plants)
├── batch_create_configs.py         # Config generator
├── config_manager.py               # Configuration manager
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
# Clone repository
git clone https://github.com/zhesun-0209/PV-Forecasting.git
cd PV-Forecasting

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

### GPU Acceleration (Automatic)

| Model | GPU Support | Library | Acceleration |
|-------|-------------|---------|--------------|
| LSTM | ✓ Auto | PyTorch | Full GPU |
| GRU | ✓ Auto | PyTorch | Full GPU |
| Transformer | ✓ Auto | PyTorch | Full GPU |
| TCN | ✓ Auto | PyTorch | Full GPU |
| XGBoost | ✓ Auto | xgboost | GPU if CUDA available |
| LightGBM | ✓ Auto | lightgbm | GPU if CUDA available |
| Random Forest | Optional | cuML/sklearn | GPU with cuML |
| Linear | Optional | cuML/sklearn | GPU with cuML |

**Performance:** 6/8 models use GPU by default, covering 240/284 experiments (85%).

---

## 📊 Applied Optimizations

This version includes several improvements over the original GitHub repository:

### 1. GPU OOM Protection ✅
- **ML models**: Batch prediction (1000 samples/batch)
- **DL models**: GPU memory clearing before training
- **Benefit**: Handle large test sets without crashes

### 2. Tree Model Parameter Optimization ✅
- **High complexity**:
  - `n_estimators`: 100 (was 200) - 2× faster
  - `max_depth`: 10 (was 15) - prevent overfitting
  - `verbosity`: -1 (silent mode)

### 3. Scenario Naming Standardization ✅
- **Enforced 6 standard types**: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+
- **No hour information** in scenario names
- **Consistent** across ML and DL models

### 4. Output Format Unification ✅
- **ASCII-only output** (Windows compatible)
- **Unified progress messages** between ML and DL
- **No Unicode errors** (✓ → [OK], ✗ → [ERROR])

### 5. Multi-Plant Support ✅
- **Batch config generation**: Scan `data/` directory automatically
- **Unified config system**: Template + plant-specific overrides
- **Batch experiments**: Run 284 experiments for each plant
- **Scalable**: Support 130+ plants

### 6. Daily Day-Ahead Prediction ✅
- **Prediction method**: One prediction per day (at 23:00)
- **No overlapping**: Each day predicted once, clean evaluation
- **Sample efficiency**: ~993 daily samples vs ~24K hourly samples
- **Training speed**: 10-16× faster than hourly sliding
- **Accuracy**: Comparable (MAE difference <5%)
- **Day-ahead focus**: Perfect alignment with operational forecasting

---

## 🎓 Configuration Management

### Plant Configuration

Each plant has a dedicated YAML file in `config/plants/`:

```yaml
# config/plants/Plant1140.yaml
plant_id: "1140"
plant_name: "Project 1140"
data_path: "data/Project1140.csv"

# Data period
start_date: "2022-01-01"
end_date: "2024-09-28"

# Data split ratios
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1

# Model parameters (can override template)
dl_params:
  low:
    d_model: 16
    num_layers: 1
    epochs: 20
  high:
    d_model: 32
    num_layers: 2
    epochs: 50

ml_params:
  low:
    n_estimators: 50
    max_depth: 5
  high:
    n_estimators: 100
    max_depth: 10
```

### Batch Config Generation

```bash
# Scan data/ directory and create configs for all CSV files
python batch_create_configs.py

# Force overwrite existing configs
python batch_create_configs.py --force

# Specify date range
python batch_create_configs.py --start-date 2022-01-01 --end-date 2024-12-31
```

Supported CSV naming patterns:
- `Project1140.csv` → Plant ID: 1140
- `plant_1141.csv` → Plant ID: 1141
- `1142.csv` → Plant ID: 1142

---

## 🔬 Experiment Design

### Total: 284 Experiments per Plant

**DL Models (160 experiments):**
- LSTM, GRU, Transformer, TCN
- 2 complexity levels (low/high)
- 2 lookback windows (24h/72h) for PV scenarios
- 2 time encoding options (enabled/disabled)
- 4 PV-based scenarios + 2 NWP-only scenarios

**ML Models (120 experiments):**
- Random Forest, XGBoost, LightGBM
- 2 complexity levels (low/high)
- 2 lookback windows (24h/72h) for PV scenarios
- 2 time encoding options (enabled/disabled)
- 4 PV-based scenarios + 2 NWP-only scenarios

**Linear Model (4 experiments):**
- Linear Regression
- 2 time encoding options
- 2 NWP-only scenarios

---

## 📈 Evaluation Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **NRMSE** (Normalized RMSE)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)

### Evaluation Strategy

**Metrics Calculation**: Based on ALL multi-step predictions
- Comprehensive assessment of model performance
- ~210,240 prediction values (8,760 samples × 24 hours)

**CSV Output**: Only 24h-ahead predictions
- Day-ahead forecasting scenario
- ~8,778 values for visualization and analysis

---

## 📊 Results Format

### Summary Results CSV

Each experiment generates a row in the results CSV:

| Column | Description |
|--------|-------------|
| experiment_name | Unique experiment identifier |
| model | Model type (LSTM, XGB, etc.) |
| complexity | low/high |
| feature_combo | Scenario (PV, PV+NWP, etc.) |
| lookback_hours | 24 or 72 |
| use_time_encoding | True/False |
| mae | Mean Absolute Error |
| rmse | Root Mean Squared Error |
| r2 | R-squared |
| train_time_sec | Training time |
| test_samples | Number of test samples |
| best_epoch | Best epoch (DL only) |
| param_count | Model parameters |

### Output Files

**Single Plant:**
- `all_experiments_results_YYYYMMDD_HHMMSS.csv` (284 rows)

**Multi-Plant:**
- `results_1140_YYYYMMDD_HHMMSS.csv` (284 rows per plant)
- `results_1141_YYYYMMDD_HHMMSS.csv`
- ...

---

## 💻 Running on Google Colab

### Quick Start

```python
# 1. Clone repository
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# 2. Install dependencies
!pip install torch torchvision torchaudio
!pip install pyyaml pandas scikit-learn xgboost lightgbm matplotlib seaborn tqdm

# 3. Optional: Install cuML for GPU Random Forest
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# 4. Restart runtime (Runtime → Restart runtime)

# 5. Run experiments
%cd PV-Forecasting
!python run_all_experiments.py
```

### Download Results

```python
from google.colab import files
files.download('all_experiments_results_*.csv')
```

### Notes
- **Execution time**: ~4-8 hours for 284 experiments
- **GPU recommended**: Tesla T4 or better
- **Free tier limit**: 12-hour sessions
- **Auto-save**: Results saved after each experiment

---

## 🔧 Advanced Usage

### Adding New Plants

1. Add CSV file to `data/` directory:
   ```
   data/
   ├── Project1140.csv
   ├── Project1141.csv  ← New plant
   └── Project1142.csv  ← New plant
   ```

2. Generate configurations:
   ```bash
   python batch_create_configs.py
   ```

3. Run multi-plant experiments:
   ```bash
   python run_experiments_multi_plant.py
   ```

### Resume Interrupted Experiments

The system automatically resumes from the last completed experiment:

```bash
# Run experiments
python run_all_experiments.py

# If interrupted, run again - will skip completed experiments
python run_all_experiments.py
```

Progress is tracked in the results CSV file.

---

## 📖 Key Findings

Based on experiments with Plant 1140:

1. **LightGBM outperforms LSTM** in high complexity:
   - LightGBM: MAE=3.32, RMSE=6.93, R²=0.896
   - LSTM: MAE=4.35, RMSE=7.81, R²=0.867

2. **Feature combination matters**:
   - PV+NWP outperforms single features
   - NWP sometimes beats NWP+ (smoothness advantage)

3. **Training efficiency**:
   - LSTM trains 2.6× faster than LightGBM
   - LightGBM provides better accuracy

4. **Prediction horizon**:
   - 24h-ahead predictions are most relevant for day-ahead forecasting
   - Multi-step evaluation shows comprehensive model performance

---

## 🎯 Optimizations Applied

This version includes 5 key optimizations over the base GitHub repository:

| Optimization | Status | Benefit |
|--------------|--------|---------|
| GPU OOM Protection | ✅ | Batch prediction + memory management |
| Tree Model Params | ✅ | n=100, d=10 (faster, less overfitting) |
| Scenario Naming | ✅ | Standardized to 6 types |
| Output Format | ✅ | ASCII-only, Windows compatible |
| Multi-Plant Support | ✅ | Batch process 130+ plants |
| 24h-Ahead Extraction | ✅ | Dual evaluation strategy |

**Shuffle functionality**: Preserved from original (as requested)

---

## 📊 Technical Details

### Data Split Strategy

The system uses random shuffle with fixed seed for reproducibility:

```python
# Default configuration
train_ratio: 0.8  # 80% for training
val_ratio: 0.1    # 10% for validation
test_ratio: 0.1   # 10% for testing
random_seed: 42   # Fixed seed for reproducibility
```

All 284 experiments use the **same data split** to ensure fair comparison.

### Prediction Method: Daily Day-Ahead Forecasting

**Daily Prediction Approach** (Aligned with paper requirements):
- **One prediction per day** (made at 23:00)
- **Input**: Previous day's 24h historical data + Next day's NWP forecast
- **Output**: Next day's full 24h generation forecast
- **No overlapping predictions**: Clean, interpretable results

**Example**:
```
Day 1 (23:00):
  Input: Day 1's 24h historical PV + Day 2's 24h NWP forecast
  Output: Day 2's 24h generation prediction (00:00-23:00)

Day 2 (23:00):
  Input: Day 2's 24h historical PV + Day 3's 24h NWP forecast
  Output: Day 3's 24h generation prediction (00:00-23:00)
```

**Data Statistics** (Plant 1140):
- Total period: 2022-01-01 to 2024-09-28 (1,002 days)
- Daily samples: ~993 days (some days incomplete)
- Training: ~794 days (80%)
- Validation: ~99 days (10%)
- Testing: ~100 days (10%)

**Evaluation**:
- Metrics calculated on ALL predictions (~100 days × 24 hours = 2,400 values)
- Each day's 24h forecast evaluated independently
- True day-ahead operational scenario

### GPU Memory Management

**ML Models**:
```python
# Batch prediction to avoid OOM
batch_size = 1000
for i in range(0, n_test, batch_size):
    batch_preds = model.predict(X_test[i:i+batch_size])
```

**DL Models**:
```python
# Clear GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

---

## 🔬 Research Context

This project supports day-ahead (24-hour) PV forecasting for:
- **Charging infrastructure planning**
- **Microgrid operations**
- **V2G (Vehicle-to-Grid) scheduling**
- **Battery storage optimization**

Accurate forecasts enable:
- Feasible charging schedules
- Cost-effective operations
- Emissions reduction
- Grid stability

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{pv-forecasting-2025,
  title={PV-Forecasting: A Multi-Model Comparison for Solar Power Prediction},
  author={Zhe Sun},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zhesun-0209/PV-Forecasting}
}
```

---

## 🐛 Troubleshooting

### Issue: GPU OOM during ML prediction
**Solution**: Already fixed with batch prediction (batch_size=1000)

### Issue: Unicode errors on Windows
**Solution**: Already fixed with ASCII-only output

### Issue: LightGBM warnings flooding console
**Solution**: Already fixed with `verbosity=-1`

### Issue: cuML installation fails
**Solution**: System falls back to sklearn CPU automatically

### Issue: Different plants have different data ranges
**Solution**: Use `batch_create_configs.py --start-date --end-date` to specify common range

---

## 📧 Contact

For questions or suggestions, please submit an Issue or Pull Request.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

**Original Repository**: [zhesun-0209/PV-Forecasting](https://github.com/zhesun-0209/PV-Forecasting)

**Optimizations**: Enhanced version with multi-plant support and production-ready features

---

**Last Updated**: 2025-10-09  
**Version**: 1.0 (Optimized)
