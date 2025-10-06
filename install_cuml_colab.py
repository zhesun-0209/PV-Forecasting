"""
Install cuML with correct dependencies on Colab
"""
import subprocess
import sys

def run_cmd(cmd):
    """Run shell command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0

print("="*60)
print("Installing cuML with correct CUDA dependencies")
print("="*60)

# Step 1: Install cuda-python with correct version
print("\nStep 1: Installing cuda-python...")
run_cmd("pip uninstall -y cuda-python")
run_cmd("pip install cuda-python==12.6.0")

# Step 2: Reinstall cuML
print("\nStep 2: Reinstalling cuML...")
run_cmd("pip install --force-reinstall --no-deps cuml-cu11")

# Step 3: Install cuML dependencies
print("\nStep 3: Installing cuML dependencies...")
deps = [
    "cudf-cu11",
    "cupy-cuda11x>=12.0.0",
    "cuvs-cu11",
    "dask-cuda",
    "dask-cudf-cu11",
    "joblib>=0.11",
    "libcuml-cu11",
    "numba",
    "numpy",
    "pylibraft-cu11",
    "raft-dask-cu11",
    "rapids-dask-dependency",
    "rmm-cu11",
    "scikit-learn>=1.5",
    "scipy>=1.8.0",
    "treelite"
]

for dep in deps:
    run_cmd(f"pip install -q {dep}")

print("\n" + "="*60)
print("Installation complete! Testing cuML...")
print("="*60)

# Test cuML
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    print("\n✓ cuML Random Forest: Available")
    print("✓ cuML Linear Regression: Available")
    print("\nAll GPU-accelerated models ready!")
except Exception as e:
    print(f"\n✗ cuML still not working: {e}")
    print("\nFalling back to CPU versions (still works, just slower)")

print("\n" + "="*60)

