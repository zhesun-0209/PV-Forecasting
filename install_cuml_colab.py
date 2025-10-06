"""
Complete cuML GPU installation for Colab
Uses system-level LD_LIBRARY_PATH configuration
"""
import subprocess
import sys
import os

def run(cmd):
    """Execute shell command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

print("="*60)
print("Installing cuML for GPU (All 8 models)")
print("="*60)

# Step 1: Clean environment
print("\n[1/7] Cleaning environment...")
run(f"{sys.executable} -m pip uninstall -y -q cupy cupy-cuda11x cupy-cuda12x cuml-cu11 cuml-cu12 cuda-python 2>/dev/null")

# Step 2: Install CUDA dependencies
print("[2/7] Installing CUDA runtime...")
run(f"{sys.executable} -m pip install -q cuda-python==11.8.7")
run(f"{sys.executable} -m pip install -q nvidia-cuda-runtime-cu11")

# Step 3: Install cuda-bindings
print("[3/7] Installing cuda-bindings...")
run(f"{sys.executable} -m pip install -q cuda-bindings~=11.8.7")

# Step 4: Install CuPy
print("[4/7] Installing CuPy...")
run(f"{sys.executable} -m pip install -q cupy-cuda11x==13.0.0")

# Step 5: Install cuML
print("[5/7] Installing cuML...")
run(f"{sys.executable} -m pip install -q cuml-cu11==25.6.*")

# Step 6: Configure LD_LIBRARY_PATH in system (critical!)
print("[6/7] Configuring system library path...")

# Write a shell script to set LD_LIBRARY_PATH and run Python
setup_script = """#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
exec python3 "$@"
"""

with open("/tmp/python_with_cuda.sh", "w") as f:
    f.write(setup_script)

os.chmod("/tmp/python_with_cuda.sh", 0o755)
print("  ✓ Created wrapper script: /tmp/python_with_cuda.sh")

# Step 7: Test in a new process with correct LD_LIBRARY_PATH
print("[7/7] Testing cuML GPU...")

test_code = """
import numpy as np
try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    
    X = np.random.rand(50, 3).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)
    
    rf = RandomForestRegressor(n_estimators=5, max_depth=3)
    rf.fit(X, y)
    print("✓ Random Forest GPU: Working")
    
    lr = LinearRegression()
    lr.fit(X, y)
    print("✓ Linear Regression GPU: Working")
    
    print("\\nSUCCESS: cuML GPU ready!")
    exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)
"""

result = subprocess.run(
    ["/tmp/python_with_cuda.sh", "-c", test_code],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode == 0:
    print("\n" + "="*60)
    print("SUCCESS: All 8 models GPU-ready!")
    print("="*60)
    print("\nTo run experiments, use:")
    print("  /tmp/python_with_cuda.sh run_all_experiments.py")
    print("\nOR create an alias:")
    print("  !echo 'alias python_cuda=\"/tmp/python_with_cuda.sh\"' >> ~/.bashrc")
    print("  Then: !python_cuda run_all_experiments.py")
else:
    print(result.stderr)
    print("\n" + "="*60)
    print("cuML GPU setup failed")
    print("="*60)
    print("\nFallback: RF and Linear will use CPU")
    print("Run normally: python run_all_experiments.py")

print("\n" + "="*60)
