#!/usr/bin/env python3
"""
Colab cuML GPU setup script - Clean installation
Resolves dependency conflicts between cu11 and cu12 packages
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run shell command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*60)
    print("Setting up cuML GPU support for Colab")
    print("="*60)
    
    # Step 1: Check CUDA version
    print("\n[1/6] Checking CUDA version...")
    run_command("nvidia-smi", "NVIDIA GPU Info")
    
    # Step 2: Uninstall ALL conflicting CuPy packages
    print("\n[2/6] Removing ALL CuPy packages to resolve conflicts...")
    cupy_packages = [
        "cupy-cuda11x",
        "cupy-cuda12x",
        "cupy",
        "cupy-cuda115",
        "cupy-cuda116",
        "cupy-cuda117",
        "cupy-cuda118",
        "cupy-cuda11",
        "cupy-cuda12"
    ]
    
    for pkg in cupy_packages:
        subprocess.run(f"pip uninstall -y {pkg}", shell=True, capture_output=True)
    
    print("All CuPy packages removed")
    
    # Step 3: Uninstall ALL cuML packages
    print("\n[3/6] Removing ALL cuML packages...")
    cuml_packages = [
        "cuml-cu11",
        "cuml-cu12",
        "cuml",
        "libcuml-cu11",
        "libcuml-cu12",
        "cudf-cu11",
        "cudf-cu12"
    ]
    
    for pkg in cuml_packages:
        subprocess.run(f"pip uninstall -y {pkg}", shell=True, capture_output=True)
    
    print("All cuML packages removed")
    
    # Step 4: Install CUDA 12 compatible packages
    print("\n[4/6] Installing CUDA 12 packages...")
    
    # Install CuPy for CUDA 12
    if not run_command(
        "pip install cupy-cuda12x --no-cache-dir",
        "Installing CuPy for CUDA 12"
    ):
        print("Warning: CuPy installation had issues, continuing...")
    
    # Install cuML for CUDA 12
    if not run_command(
        "pip install cuml-cu12==25.8.0 --extra-index-url=https://pypi.nvidia.com --no-cache-dir",
        "Installing cuML for CUDA 12"
    ):
        print("Warning: cuML installation had issues, continuing...")
    
    # Step 5: Verify installation
    print("\n[5/6] Verifying cuML installation...")
    
    verify_code = """
import sys
try:
    # Test cuML import
    import cuml
    print(f"SUCCESS: cuML version {cuml.__version__}")
    
    # Test cuML RandomForest
    from cuml.ensemble import RandomForestRegressor
    print("SUCCESS: cuML RandomForestRegressor imported")
    
    # Test cuML Linear
    from cuml.linear_model import LinearRegression
    print("SUCCESS: cuML LinearRegression imported")
    
    # Test basic functionality
    import numpy as np
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.rand(100, 1).astype(np.float32)
    
    # Test RF
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    rf.fit(X, y.ravel())
    print("SUCCESS: RandomForest training works")
    
    # Test Linear
    lr = LinearRegression()
    lr.fit(X, y)
    print("SUCCESS: LinearRegression training works")
    
    print("\\n" + "="*60)
    print("cuML GPU setup completed successfully!")
    print("="*60)
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    with open('/tmp/test_cuml.py', 'w') as f:
        f.write(verify_code)
    
    success = run_command(
        "python /tmp/test_cuml.py",
        "Testing cuML functionality"
    )
    
    # Step 6: Instructions
    print("\n[6/6] Setup Summary")
    print("="*60)
    
    if success:
        print("SUCCESS: cuML GPU support is ready!")
        print("\nYou can now:")
        print("  1. Clone your repository: !git clone https://github.com/zhesun-0209/PV-Forecasting.git")
        print("  2. Run experiments: !python run_all_experiments.py")
    else:
        print("PARTIAL SUCCESS: Some tests failed, but cuML may still work")
        print("\nTroubleshooting:")
        print("  1. Restart Colab runtime: Runtime > Restart runtime")
        print("  2. Re-run this script")
        print("  3. Check GPU availability: !nvidia-smi")
    
    print("="*60)

if __name__ == "__main__":
    main()

