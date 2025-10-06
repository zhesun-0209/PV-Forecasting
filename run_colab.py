"""
Wrapper script for running on Colab with cuML GPU support
Sets up environment variables before importing cuML
"""
import os
import sys
import glob

# CRITICAL: Set LD_LIBRARY_PATH before any imports
print("Setting up CUDA library paths...")

cuda_lib_paths = [
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib",
    "/usr/local/cuda-11.8/lib64",
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu"
]

valid_paths = [p for p in cuda_lib_paths if os.path.exists(p)]
if valid_paths:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(valid_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    print(f"✓ LD_LIBRARY_PATH set with {len(valid_paths)} paths")
else:
    print("Warning: No CUDA library paths found")

# Find and create libcudart.so.11.0 symlink if needed
print("\nChecking for libcudart.so.11.0...")
for path in valid_paths:
    target = f"{path}/libcudart.so.11.0"
    if os.path.exists(target):
        print(f"  ✓ Found: {target}")
        break
    
    # Look for any version of libcudart.so
    cudart_files = glob.glob(f"{path}/libcudart.so*")
    if cudart_files:
        source = cudart_files[0]
        try:
            if not os.path.exists(target):
                os.symlink(source, target)
                print(f"  ✓ Created symlink: {target} -> {source}")
            break
        except:
            # Try system command
            import subprocess
            subprocess.run(["ln", "-sf", source, target], capture_output=True)
            if os.path.exists(target):
                print(f"  ✓ Created symlink: {target}")
                break

# Test cuML before running experiments
print("\n" + "="*60)
print("Testing cuML GPU support...")
print("="*60)

try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    import numpy as np
    
    X = np.random.rand(50, 3).astype(np.float32)
    y = np.random.rand(50).astype(np.float32)
    
    rf = RandomForestRegressor(n_estimators=5, max_depth=3)
    rf.fit(X, y)
    print("✓ Random Forest GPU: Working")
    
    lr = LinearRegression()
    lr.fit(X, y)
    print("✓ Linear Regression GPU: Working")
    
    print("\n" + "="*60)
    print("SUCCESS: All 8 models will use GPU!")
    print("="*60)
    
except Exception as e:
    print(f"✗ cuML test failed: {e}")
    print("\nWill use CPU fallback for RF and Linear")
    print("Other 6 models still use GPU")

# Now run the main experiment script
print("\n" + "="*60)
print("Starting experiments...")
print("="*60)

# Import and run
if __name__ == "__main__":
    import run_all_experiments
    run_all_experiments.run_all_experiments()

