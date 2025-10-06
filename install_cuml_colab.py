"""
Install cuML with correct dependencies on Colab
Fixed version for CUDA 11 compatibility
"""
import subprocess
import sys

def run_cmd(cmd):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"Warning: {result.stderr[:200]}")
    return result.returncode == 0

print("="*60)
print("Installing cuML for GPU-accelerated ML models")
print("="*60)

# Check CUDA version
print("\nChecking CUDA version...")
run_cmd("nvcc --version | grep release")

# Method 1: Try direct conda install (fastest)
print("\nAttempting conda installation...")
conda_success = run_cmd("conda install -c rapidsai -c conda-forge -c nvidia cuml=25.6 python=3.12 cudatoolkit=11.8 -y")

if not conda_success:
    print("\nConda installation failed, trying pip method...")
    
    # Method 2: Use pip with specific version
    print("\nStep 1: Ensuring cuda-python 11.8.x...")
    run_cmd("pip install cuda-python>=11.8,<12.0")
    
    print("\nStep 2: Installing cuML with CUDA 11...")
    # Install the complete RAPIDS stack for CUDA 11
    run_cmd("pip install --extra-index-url=https://pypi.nvidia.com cuml-cu11==25.6.*")

# Test installation
print("\n" + "="*60)
print("Testing cuML installation...")
print("="*60)

try:
    import cuml
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    import numpy as np
    
    # Quick functionality test
    print("\n✓ cuML imported successfully")
    
    # Test Random Forest
    X_test = np.random.rand(100, 5)
    y_test = np.random.rand(100)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    rf.fit(X_test, y_test)
    print("✓ Random Forest GPU: Working")
    
    # Test Linear Regression
    lr = LinearRegression()
    lr.fit(X_test, y_test)
    print("✓ Linear Regression GPU: Working")
    
    print("\n" + "="*60)
    print("SUCCESS: All GPU-accelerated ML models ready!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ cuML test failed: {e}")
    print("\nDon't worry! The code will automatically use CPU fallback.")
    print("Only Random Forest and Linear will be slower.")
    print("All other models (LSTM, GRU, Transformer, TCN, XGBoost, LightGBM) still use GPU.")
    print("\n" + "="*60)
