"""
Automatically detect CUDA version and install matching cuML
"""
import subprocess
import sys
import re

def run(cmd):
    """Execute command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

print("="*60)
print("Auto-detecting CUDA and installing cuML")
print("="*60)

# Detect CUDA version
print("\n[1] Detecting CUDA version...")
nvcc_output = run("nvcc --version")
print(nvcc_output)

cuda_version = None
match = re.search(r'release (\d+)\.(\d+)', nvcc_output)
if match:
    major, minor = match.groups()
    cuda_version = f"{major}.{minor}"
    print(f"  ✓ CUDA version: {cuda_version}")
else:
    print("  ✗ Could not detect CUDA version")
    print("  Assuming CUDA 12 (Colab default)")
    cuda_version = "12.0"

# Determine cuML package
cuda_major = int(cuda_version.split('.')[0])

if cuda_major >= 12:
    cuml_package = "cuml-cu12"
    print(f"\n[2] Using cuML for CUDA 12")
elif cuda_major == 11:
    cuml_package = "cuml-cu11"
    print(f"\n[2] Using cuML for CUDA 11")
else:
    print(f"\n[2] Unsupported CUDA version: {cuda_version}")
    sys.exit(1)

# Clean previous installations
print(f"\n[3] Cleaning previous installations...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", 
                "cuml-cu11", "cuml-cu12", "cupy-cuda11x", "cupy-cuda12x"], capture_output=True)

# Install matching cuML
print(f"\n[4] Installing {cuml_package}...")
result = subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    f"{cuml_package}>=25.0"
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"  ✓ {cuml_package} installed")
else:
    print(f"  ✗ Installation failed")
    print(result.stderr[:500])

# Test cuML
print(f"\n[5] Testing cuML GPU...")
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    import cudf
    import numpy as np
    
    print("  ✓ cuML modules imported")
    
    # Test with cuDF data
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)
    
    # Test RF
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    rf.fit(X_cudf, y_cudf)
    pred = rf.predict(X_cudf[:10])
    print(f"  ✓ Random Forest GPU working")
    
    # Test Linear
    lr = LinearRegression()
    lr.fit(X_cudf, y_cudf)
    pred = lr.predict(X_cudf[:10])
    print(f"  ✓ Linear Regression GPU working")
    
    print("\n" + "="*60)
    print("SUCCESS: cuML GPU fully functional!")
    print("="*60)
    print(f"\nInstalled: {cuml_package}")
    print("All 8 models will use GPU!")
    
except Exception as e:
    print(f"  ✗ cuML test failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("cuML GPU not working - using CPU fallback")
    print("="*60)
    print("6/8 models will use GPU (DL + XGBoost + LightGBM)")

print("\n" + "="*60)

