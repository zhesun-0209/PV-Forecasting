# Running on Google Colab

## Quick Start (Recommended)

### Step 1: Open a new Colab notebook

### Step 2: Clone and setup
```python
# Clone repository
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# Install base dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q pyyaml pandas scikit-learn xgboost lightgbm matplotlib seaborn tqdm

# Setup cuML GPU support (automatic CUDA version detection)
!python setup_cuml_gpu.py

# IMPORTANT: After setup_cuml_gpu.py completes, you MUST restart the runtime
# Go to: Runtime → Restart runtime
```

### Step 3: After runtime restart, run experiments
```python
%cd PV-Forecasting
!python run_all_experiments.py
```

## Alternative: Manual cuML Installation

If `setup_cuml_gpu.py` fails, try manual installation:

```python
# Check CUDA version
!nvcc --version

# For CUDA 12.x
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# For CUDA 11.x
!pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com

# Verify installation
!python -c "import cuml; print('cuML version:', cuml.__version__)"

# RESTART RUNTIME after installation
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cuda.bindings.cyruntime'`

**Solution:**
```python
# Restart runtime first: Runtime → Restart runtime
# Then reinstall:
!pip uninstall -y cupy-cuda11x cupy-cuda12x
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
# Restart runtime again
```

### Issue: Dependency conflicts between cu11 and cu12

**Solution:**
```python
# Uninstall all RAPIDS packages
!pip uninstall -y cuml-cu11 cuml-cu12 cudf-cu11 cudf-cu12
# Reinstall only cu12
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
# Restart runtime
```

### Issue: GPU not detected

**Check GPU:**
```python
!nvidia-smi
```

**Verify PyTorch GPU:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Expected Output

After successful setup, you should see:

```
cuML available (GPU-accelerated Random Forest and Linear Regression)
XGBoost GPU available
LightGBM GPU available
================================================================================
PV Forecasting: Running 284 Experiments
================================================================================
GPU: Tesla T4 (or similar)
```

## Results

Experiment results will be saved to:
- `all_experiments_results_YYYYMMDD_HHMMSS.csv`

Download results:
```python
from google.colab import files
files.download('all_experiments_results_*.csv')
```

## Notes

1. **Runtime Restart Required**: cuML installation requires a runtime restart to take effect
2. **Execution Time**: 284 experiments may take 4-8 hours depending on GPU
3. **Free Tier Limits**: Colab free tier has 12-hour session limits
4. **Results Auto-Save**: Each experiment result is saved immediately after completion

